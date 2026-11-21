"""LangGraph workflow: rewrite (conversation memory) -> route -> retrieve and/or SQL ->
grounded synthesis with citations -> citation/number check."""
from __future__ import annotations

import re
import time
from typing import Literal, TypedDict

from langgraph.graph import END, StateGraph

from . import config, llm, sql_agent
from .index import get_index

REFUSAL = "I don't have enough information in the filings or the warehouse to answer that."


class State(TypedDict, total=False):
    question: str
    history: list[dict]            # [{"role": "user"|"assistant", "content": str}]
    allowed_tickers: list[str] | None
    strategy: str                  # chunk set: "semantic" | "fixed"
    standalone: str
    route: Literal["doc", "sql", "hybrid", "none"]      # path actually taken
    llm_route: str                                      # the router model's own label
    chunks: list[dict]
    sql: sql_agent.SQLResult | None
    answer: str
    citations: list[dict]
    refused: bool
    unsupported_numbers: list[str]
    trace: list[dict]              # per-node latency and tokens


def _log(state: State, node: str, t0: float, tokens: int = 0) -> list[dict]:
    return state.get("trace", []) + [{"node": node, "ms": round((time.perf_counter() - t0) * 1000),
                                      "tokens": tokens}]


# ---- nodes -------------------------------------------------------------------------------
def rewrite(state: State) -> State:
    """Turn a follow-up ("what about Lowe's?") into a standalone question."""
    t0, hist = time.perf_counter(), state.get("history") or []
    if not hist:
        return {"standalone": state["question"], "trace": _log(state, "rewrite", t0)}
    convo = "\n".join(f"{m['role']}: {m['content'][:400]}" for m in hist[-6:])
    out = llm.chat([{"role": "user", "content":
                     f"Conversation:\n{convo}\n\nFollow-up: {state['question']}\n\n"
                     "Rewrite the follow-up as one standalone question that names the company and "
                     "fiscal year if the conversation implies them. Reply with the question only."}],
                   max_tokens=80)
    q = out.text.strip().strip('"') or state["question"]
    return {"standalone": q, "trace": _log(state, "rewrite", t0, out.prompt_tokens + out.completion_tokens)}


ROUTER = """Route a question about six US retailers (Walmart, Target, Costco, Home Depot, Lowe's, Best Buy)
and a UK online retail transactions dataset.
- "sql": answered by numbers in a financials table (revenue, cost of sales, gross profit, SG&A, operating
  income, net income, diluted EPS, assets, inventory, cash, equity, debt, operating cash flow, capex,
  buybacks, dividends by company and fiscal year) or by the online retail transactions.
- "doc": answered by the text of 10-K reports (strategy, risks, store counts, segments, employees,
  explanations, management commentary, cybersecurity, properties, legal matters).
- "hybrid": needs both a financial figure AND text from the 10-K. Any question with two parts where one
  asks for a number and the other asks what drove it, why, how the company describes something, or what
  it says/plans is "hybrid", never "sql".
- "none": about something else entirely (other companies, stock prices, the future, private matters).
Reply with JSON only: {"route": "sql" | "doc" | "hybrid" | "none"}"""


_SCOPE = re.compile("|".join(re.escape(a) for al in config.TICKERS.values() for a in al)
                    + r"|online retail|\b(?:wmt|tgt|cost|hd|low|bby)\b", re.I)


def in_scope(question: str) -> bool:
    """Does the question name one of the six retailers or the online retail dataset?"""
    return bool(_SCOPE.search(question))


def route(state: State) -> State:
    t0 = time.perf_counter()
    data, out = llm.chat_json([{"role": "system", "content": ROUTER},
                               {"role": "user", "content": state["standalone"]}], max_tokens=20)
    r = (data or {}).get("route", "hybrid")
    r = llm_route = r if r in ("doc", "sql", "hybrid", "none") else "hybrid"
    # The LLM's route is advisory. v1 trusted it outright and lost 27% of answerable questions to
    # "none"/"sql" misroutes, so: refuse up front only when the question names nothing in scope,
    # and never skip the 10-K search (the synthesizer refuses if the evidence is weak).
    if r == "none" and in_scope(state["standalone"]):
        r = "doc"
    elif r == "sql":
        r = "hybrid"
    return {"route": r, "llm_route": llm_route, "trace": _log(state, "route", t0, out.prompt_tokens + out.completion_tokens)}


def retrieve(state: State) -> State:
    t0 = time.perf_counter()
    ix = get_index(state.get("strategy", "semantic"))
    chunks = ix.search(state["standalone"], k=config.CONTEXT_K, mode="hybrid_rerank",
                       allowed_tickers=state.get("allowed_tickers"))
    # keep only chunks the reranker considers related to the question
    chunks = [c for c in chunks if c["score"] is None or c["score"] >= config.MIN_RERANK_SCORE]
    return {"chunks": chunks, "trace": _log(state, "retrieve", t0)}


def run_sql(state: State) -> State:
    t0 = time.perf_counter()
    allowed = state.get("allowed_tickers")
    res = sql_agent.run(state["standalone"])
    if res.ok and allowed is not None and "ticker" in res.columns:  # access control on rows
        i = res.columns.index("ticker")
        res.rows = [r for r in res.rows if r[i] in allowed]
    return {"sql": res, "trace": _log(state, "sql", t0, res.tokens)}


SYNTH = """You are an analyst assistant. Answer the question using ONLY the sources below.
- Cite every factual sentence with the source label in brackets, e.g. [1] or [SQL].
- Copy numbers exactly from the sources; you may round billions to two decimals.
- If the sources do not contain the answer, reply exactly: "{refusal}"
- When a 10-K source states the exact figure asked about, use it; use [SQL] for the standard metrics
  it computes. Ignore sources that are about a different company, year or metric.
- Be concise: at most 5 sentences."""


def synthesize(state: State) -> State:
    t0 = time.perf_counter()
    chunks, sql = state.get("chunks") or [], state.get("sql")
    has_sql = bool(sql and sql.ok and sql.rows and any(v is not None for row in sql.rows for v in row))
    if state.get("route") == "none" or (not chunks and not has_sql):
        return {"answer": REFUSAL, "refused": True, "citations": [], "unsupported_numbers": [],
                "trace": _log(state, "synthesize", t0)}
    sources = [f"[{i}] {c['company']} FY{c['fiscal_year']} 10-K, Item {c['item']} {c['section']}:\n{c['text']}"
               for i, c in enumerate(chunks, 1)]
    if has_sql:
        sources.append("[SQL] " + sql_agent.as_text(sql))
    out = llm.chat([{"role": "system", "content": SYNTH.format(refusal=REFUSAL)},
                    {"role": "user", "content": "Sources:\n\n" + "\n\n".join(sources) +
                     f"\n\nQuestion: {state['standalone']}"}], max_tokens=350)
    answer = out.text
    refused = REFUSAL.lower()[:40] in answer.lower()
    labels = [x.strip() for grp in re.findall(r"\[([\dSQL,\s]+)\]", answer) for x in grp.split(",")]
    cited = {int(n) for n in labels if n.isdigit() and 0 < int(n) <= len(chunks)}
    citations = [{"label": n, "chunk_id": chunks[n - 1]["chunk_id"], "company": chunks[n - 1]["company"],
                  "fiscal_year": chunks[n - 1]["fiscal_year"], "section": chunks[n - 1]["section"],
                  "url": chunks[n - 1]["source_url"]} for n in sorted(cited)]
    if has_sql and "SQL" in labels:
        citations.append({"label": "SQL", "sql": sql.sql})
    source_text = " ".join(sources)
    return {"answer": answer, "refused": refused, "citations": citations,
            "unsupported_numbers": [] if refused else unsupported_numbers(answer, source_text),
            "trace": _log(state, "synthesize", t0, out.prompt_tokens + out.completion_tokens)}


# ---- numeric grounding check ----------------------------------------------------------------
NUM = re.compile(r"\d[\d,]*(?:\.\d+)?")


def _variants(x: float) -> set[str]:
    """Spellings a source number can take once the model rounds or rescales it."""
    out = set()
    for scale in (1, 1e3, 1e6, 1e9, 0.01):
        y = x / scale
        out |= {f"{y:.0f}", f"{y:.1f}", f"{y:.2f}"}
    return out


def unsupported_numbers(answer: str, sources: str) -> list[str]:
    """Numbers in the answer that cannot be traced to any number in the sources."""
    src: set[str] = set()
    for m in NUM.findall(sources):
        try:
            src |= _variants(float(m.replace(",", "")))
        except ValueError:
            pass
    bad = []
    for m in NUM.findall(re.sub(r"\[(\d+|SQL)\]", " ", answer)):
        v = m.replace(",", "").rstrip(".")
        if not v or len(v) == 1 or re.fullmatch(r"20[12]\d", v):  # single digits, years
            continue
        x = float(v)
        if not ({f"{x:.0f}", f"{x:.1f}", f"{x:.2f}"} & src):
            bad.append(m)
    return bad


# ---- graph ------------------------------------------------------------------------------------
def _after_route(state: State) -> str:
    return {"doc": "retrieve", "hybrid": "retrieve", "sql": "sql", "none": "synthesize"}[state["route"]]


def _after_retrieve(state: State) -> str:
    return "sql" if state["route"] == "hybrid" else "synthesize"


def build_graph():
    g = StateGraph(State)
    g.add_node("rewrite", rewrite)
    g.add_node("route", route)
    g.add_node("retrieve", retrieve)
    g.add_node("sql", run_sql)
    g.add_node("synthesize", synthesize)
    g.set_entry_point("rewrite")
    g.add_edge("rewrite", "route")
    g.add_conditional_edges("route", _after_route, ["retrieve", "sql", "synthesize"])
    g.add_conditional_edges("retrieve", _after_retrieve, ["sql", "synthesize"])
    g.add_edge("sql", "synthesize")
    g.add_edge("synthesize", END)
    return g.compile()


_graph = None


def ask(question: str, history: list[dict] | None = None, allowed_tickers: list[str] | None = None,
        strategy: str = "semantic", force_route: str | None = None) -> State:
    global _graph
    if _graph is None:
        _graph = build_graph()
    state: State = {"question": question, "history": history or [], "allowed_tickers": allowed_tickers,
                    "strategy": strategy, "trace": []}
    if force_route:  # used by evaluation to score each branch on its own
        state.update(standalone=question, route=force_route)
        for node in (["retrieve"] if force_route in ("doc", "hybrid") else []) + \
                    (["sql"] if force_route in ("sql", "hybrid") else []) + ["synthesize"]:
            state.update({"retrieve": retrieve, "sql": run_sql, "synthesize": synthesize}[node](state))
        return state
    return _graph.invoke(state)
