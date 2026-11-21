"""Evaluation harness.

  python -m eval.run_eval retrieval   # chunking x retriever grid: Recall@k, MRR, latency (no LLM)
  python -m eval.run_eval sql         # text-to-SQL execution accuracy, first try vs with retry
  python -m eval.run_eval e2e         # full agent vs naive RAG baseline: routing, correctness,
                                      # faithfulness (LLM judge + numeric check), refusals
Results are written to eval/results/*.json and summarised in eval/results/summary.md.
"""
from __future__ import annotations

import json
import re
import statistics
import sys
import time
from pathlib import Path

from copilot import config, graph, llm, sql_agent
from copilot.index import MODES, Filters, get_index

GOLD = config.ROOT / "eval" / "golden.jsonl"
OUT = config.ROOT / "eval" / "results"


def load_gold() -> list[dict]:
    return [json.loads(line) for line in GOLD.open()]


def save(name: str, obj) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{name}.json").write_text(json.dumps(obj, indent=2, default=str))


# ---- relevance: the gold evidence sentence is inside the chunk -------------------------------
def norm(s: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9%$. ]", " ", s.lower())).strip()


def contains_evidence(chunk_text: str, evidence: str) -> bool:
    ev, ch = norm(evidence), norm(chunk_text)
    if ev in ch:
        return True
    toks = ev.split()  # sentence split across a chunk boundary: require 80% of its tokens
    chs = set(ch.split())
    return len(toks) > 0 and sum(t in chs for t in toks) / len(toks) >= 0.8


def first_relevant_rank(results: list[dict], q: dict) -> int | None:
    for r, c in enumerate(results, 1):
        if c["ticker"] == q["ticker"] and c["fiscal_year"] == q["fiscal_year"] and \
                contains_evidence(c["text"], q["evidence"]):
            return r
    return None


def eval_retrieval() -> dict:
    qs = [q for q in load_gold() if q["type"] == "doc"]
    out = {"n_questions": len(qs), "grid": []}
    for strategy in ("fixed", "semantic"):
        ix = get_index(strategy)
        ix.search("warm up", k=3, mode="hybrid_rerank")
        for mode in MODES:
            for auto_filters in (False, True):
                ranks, lat = [], []
                for q in qs:
                    t0 = time.perf_counter()
                    res = ix.search(q["question"], k=10, mode=mode, auto_filters=auto_filters,
                                    filters=None if auto_filters else Filters())
                    lat.append((time.perf_counter() - t0) * 1000)
                    ranks.append(first_relevant_rank(res, q))
                row = {"chunking": strategy, "retriever": mode, "metadata_filters": auto_filters,
                       "recall@1": sum(r is not None and r <= 1 for r in ranks) / len(qs),
                       "recall@5": sum(r is not None and r <= 5 for r in ranks) / len(qs),
                       "recall@10": sum(r is not None for r in ranks) / len(qs),
                       "mrr@10": sum(1 / r for r in ranks if r) / len(qs),
                       "p50_ms": statistics.median(lat),
                       "p95_ms": sorted(lat)[int(0.95 * (len(lat) - 1))]}
                out["grid"].append(row)
                print(json.dumps({k: round(v, 3) if isinstance(v, float) else v for k, v in row.items()}),
                      flush=True)
    save("retrieval", out)
    return out


# ---- SQL execution accuracy -----------------------------------------------------------------
def _match(a, b) -> bool:
    if isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool):
        close = lambda x, y: abs(x - y) <= 1e-3 * max(abs(x), abs(y), 1e-9)  # noqa: E731
        return close(a, b) or close(a * 100, b) or close(a, b * 100)  # ratio vs percent
    return str(a).strip().lower() == str(b).strip().lower()


def result_matches(gold_rows: list[tuple], pred_rows: list[tuple]) -> bool:
    """Gold values must appear, in order, among the predicted values (extra columns such as
    ticker or fiscal_year are fine)."""
    pred = [v for row in pred_rows for v in row]
    i = 0
    for g in (v for row in gold_rows for v in row):
        while i < len(pred) and not _match(g, pred[i]):
            i += 1
        if i == len(pred):
            return False
        i += 1
    return True


def eval_sql() -> dict:
    qs = [q for q in load_gold() if q["type"] == "sql"]
    con = sql_agent.connect()
    rows = []
    for q in qs:
        gold = con.execute(q["gold_sql"]).fetchall()
        t0 = time.perf_counter()
        res = sql_agent.run(q["question"])
        ok = res.ok and result_matches(gold, res.rows)
        rows.append({"id": q["id"], "question": q["question"], "correct": ok, "attempts": res.attempts,
                     "sql": res.sql, "error": res.error, "seconds": time.perf_counter() - t0})
        print(q["id"], ok, res.attempts, res.sql, flush=True)
    n = len(rows)
    out = {"n_questions": n,
           "execution_accuracy": sum(r["correct"] for r in rows) / n,
           "first_attempt_accuracy": sum(r["correct"] and r["attempts"] == 1 for r in rows) / n,
           "recovered_by_retry": sum(r["correct"] and r["attempts"] > 1 for r in rows),
           "valid_sql_rate": sum(r["error"] is None for r in rows) / n,
           "rows": rows}
    save("sql", out)
    return out


# ---- end to end ----------------------------------------------------------------------------
JUDGE_CORRECT = """Question: {q}
Reference answer: {ref}
Model answer: {ans}
Does the model answer state the reference answer's key fact (same meaning; numbers may be rounded)?
Extra correct detail is fine. Reply with JSON only: {{"correct": true | false}}"""

JUDGE_FAITHFUL = """You are checking an answer against its sources.
Notes: tickers WMT=Walmart, TGT=Target, COST=Costco, HD=Home Depot, LOW=Lowe's, BBY=Best Buy. An [SQL]
source's WHERE clause tells you which company and fiscal year its result row is for. Rounding
(713,163,000,000 -> 713.16 billion) and simple arithmetic on source values count as supported.

Sources:
{src}

Answer: {ans}

First list each factual claim in the answer and quote the source text that supports it, or write
NONE. Then on the last line reply with JSON only: {{"faithful": true}} if every claim has support,
otherwise {{"faithful": false}}."""
# v1 asked for the verdict directly; on hand-made checks it rejected even a verbatim-supported SQL
# answer. Quote-then-verdict scored 7/8 on the same checks (eval/judge_calibration.py).


def judge_faithful(src: str, ans: str) -> bool | None:
    out = llm.chat([{"role": "user", "content": JUDGE_FAITHFUL.format(src=src, ans=ans)}], max_tokens=500)
    m = re.findall(r'"faithful"\s*:\s*(true|false)', out.text)
    return None if not m else m[-1] == "true"


def judge(prompt: str, key: str) -> bool | None:
    data, _ = llm.chat_json([{"role": "user", "content": prompt}], max_tokens=200)
    return None if data is None or key not in data else bool(data[key])


def numbers_present(answer: str, gold_rows: list[tuple]) -> bool:
    """Every gold number appears in the answer (any rounding/scale the model might use)."""
    nums = [v for row in gold_rows for v in row if isinstance(v, (int, float))]
    found = {f for m in graph.NUM.findall(answer) for f in graph._variants(float(m.replace(",", "")))}
    return all(({f"{x:.0f}", f"{x:.1f}", f"{x:.2f}"} | graph._variants(x)) & found for x in nums) if nums else True


def naive_rag(question: str) -> dict:
    """Baseline: dense top-k over fixed-size chunks, no filters/rerank/SQL, plain prompt."""
    ix = get_index("fixed")
    chunks = ix.search(question, k=config.CONTEXT_K, mode="dense", auto_filters=False)
    ctx = "\n\n".join(c["text"] for c in chunks)
    out = llm.chat([{"role": "user", "content": f"Context:\n{ctx}\n\nQuestion: {question}\nAnswer concisely."}],
                   max_tokens=350)
    return {"answer": out.text, "chunks": chunks, "refused": False,
            "tokens": out.prompt_tokens + out.completion_tokens}


def eval_e2e(systems: tuple[str, ...] = ("agent", "naive_rag"), tag: str = "") -> dict:
    gold = load_gold()
    con = sql_agent.connect()
    results = {}
    for system in systems:
        rows = []
        for q in gold:
            t0 = time.perf_counter()
            if system == "agent":
                st = graph.ask(q["question"])
                ans, chunks, refused = st["answer"], st.get("chunks") or [], st.get("refused", False)
                sql = st.get("sql")
                src = "\n\n".join(f"[{i}] {c['text']}" for i, c in enumerate(chunks, 1))
                if sql is not None and sql.ok:  # first, so the length cap below never cuts it
                    src = "[SQL] " + sql_agent.as_text(sql) + "\n\n" + src
                route, tokens = st.get("llm_route", st.get("route")), sum(t["tokens"] for t in st.get("trace", []))
                unsupported_nums = st.get("unsupported_numbers", [])
            else:
                st = naive_rag(q["question"])
                ans, chunks, refused, route, tokens = st["answer"], st["chunks"], False, None, st["tokens"]
                src = "\n\n".join(f"[{i}] {c['text']}" for i, c in enumerate(chunks, 1))
                unsupported_nums = graph.unsupported_numbers(ans, src)
            secs = time.perf_counter() - t0
            refused = refused or graph.REFUSAL.lower()[:40] in ans.lower()
            row = {"id": q["id"], "type": q["type"], "question": q["question"], "answer": ans,
                   "route": route, "refused": refused, "seconds": secs, "tokens": tokens,
                   "unsupported_numbers": unsupported_nums}
            if q["type"] == "unanswerable":
                row["correct"] = refused
            elif refused:
                row["correct"] = False
            elif q["type"] == "doc":
                row["correct"] = judge(JUDGE_CORRECT.format(q=q["question"], ref=q["answer"], ans=ans), "correct")
            else:  # sql / hybrid: the gold figure must appear in the answer
                row["correct"] = numbers_present(ans, con.execute(q["gold_sql"]).fetchall())
            if not refused and q["type"] != "unanswerable":
                row["faithful"] = judge_faithful(src[:16000], ans)
            elif not refused:  # answered an unanswerable question
                row["faithful"] = False
            rows.append(row)
            print(system, q["id"], route, row.get("correct"), row.get("faithful"), f"{secs:.1f}s", flush=True)
        results[system] = summarize(rows)
        results[system]["rows"] = rows
        save(f"e2e_{system}{tag}", results[system])
    return results


def summarize(rows: list[dict]) -> dict:
    def rate(rs, key):
        vals = [r[key] for r in rs if r.get(key) is not None]
        return round(sum(vals) / len(vals), 4) if vals else None

    expected = {"doc": "doc", "sql": "sql", "hybrid": "hybrid", "unanswerable": "none"}
    by_type = {t: {"n": len(rs), "correct": rate(rs, "correct"), "faithful": rate(rs, "faithful")}
               for t in ("doc", "sql", "hybrid", "unanswerable") if (rs := [r for r in rows if r["type"] == t])}
    answered = [r for r in rows if not r["refused"]]
    answerable = [r for r in rows if r["type"] != "unanswerable"]
    lat = sorted(r["seconds"] for r in rows)
    return {
        "n": len(rows),
        "by_type": by_type,
        "answer_accuracy_answerable": rate(answerable, "correct"),
        "faithfulness_answered": rate(answered, "faithful"),
        "unsupported_answer_rate": round(1 - rate(answered, "faithful"), 4) if answered else None,
        "answers_with_untraceable_numbers": round(sum(bool(r["unsupported_numbers"]) for r in answered)
                                                  / max(len(answered), 1), 4),
        "correct_refusals": rate([r for r in rows if r["type"] == "unanswerable"], "correct"),
        "false_refusals": round(sum(r["refused"] for r in answerable) / len(answerable), 4),
        "routing_accuracy": round(sum(r["route"] == expected[r["type"]] for r in rows) / len(rows), 4)
        if rows[0]["route"] is not None else None,
        "p50_seconds": round(statistics.median(lat), 2),
        "p95_seconds": round(lat[int(0.95 * (len(lat) - 1))], 2),
        "mean_tokens": round(statistics.mean(r["tokens"] for r in rows)),
    }


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "retrieval"
    if what == "e2e":
        eval_e2e(tuple(sys.argv[2].split(",")) if len(sys.argv) > 2 else ("agent", "naive_rag"),
                 sys.argv[3] if len(sys.argv) > 3 else "")
    else:
        {"retrieval": eval_retrieval, "sql": eval_sql}[what]()
