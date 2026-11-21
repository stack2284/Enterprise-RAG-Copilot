"""Draft document questions for the golden set (step 1 of 2; step 2 is manual review).

Paragraphs are sampled from the parsed 10-K sections, independent of either chunker, and
the local LLM writes one paraphrased question per paragraph plus a verbatim evidence
sentence. Drafts whose evidence is not found verbatim in the paragraph are dropped.
Output: eval/drafts_doc.jsonl  (reviewed questions go into eval/golden.jsonl)
"""
from __future__ import annotations

import json
import random
import re

from copilot import config, ingest, llm

ITEMS = {"1", "1A", "1C", "2", "5", "7", "7A"}
PER_FILING = 9
PROMPT = """You write evaluation questions for a question-answering system over 10-K annual reports.
Company: {company}. Fiscal year: {fy}. Section: Item {item} {section}.

Passage:
\"\"\"{para}\"\"\"

Write ONE question an analyst would ask whose answer is a specific fact stated in this passage.
Rules:
- The question must name the company and the fiscal year, and must make sense without seeing the passage.
- Paraphrase: do not copy long phrases from the passage.
- The answer must be short (a number, name or one clause).
- "evidence" must be ONE sentence copied exactly, character for character, from the passage.
Reply with JSON only: {{"question": "...", "answer": "...", "evidence": "..."}}"""


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9%$. ]", " ", s.lower())).strip()


def candidate_paragraphs() -> list[dict]:
    manifest = json.loads((config.FILINGS / "manifest.json").read_text())
    out = []
    for doc in manifest:
        lines = ingest.html_to_lines((config.FILINGS / doc["file"]).read_bytes())
        cands = []
        for item, title, sec in ingest.split_sections(lines):
            if item not in ITEMS:
                continue
            for p in ingest.paragraphs(sec):
                n = len(p.split())
                # factual prose: has a number, is a real paragraph, not a table dump
                if 40 <= n <= 160 and re.search(r"\d", p) and p.count("|") < 3:
                    cands.append({"ticker": doc["ticker"], "company": doc["company"],
                                  "fiscal_year": doc["fiscal_year"], "item": item,
                                  "section": title, "para": p})
        out += random.Random(doc["file"]).sample(cands, min(PER_FILING, len(cands)))
    return out


def main() -> None:
    kept = 0
    with open(config.ROOT / "eval" / "drafts_doc.jsonl", "w") as f:
        for i, c in enumerate(candidate_paragraphs()):
            msg = PROMPT.format(company=c["company"], fy=c["fiscal_year"], item=c["item"],
                                section=c["section"], para=c["para"])
            d, _ = llm.chat_json([{"role": "user", "content": msg}], max_tokens=250)
            if not d or not all(k in d for k in ("question", "answer", "evidence")):
                continue
            if len(norm(d["evidence"])) < 20 or norm(d["evidence"]) not in norm(c["para"]):
                continue
            row = {"id": f"doc-{i:03d}", "type": "doc", **{k: c[k] for k in ("ticker", "fiscal_year", "item")},
                   "question": d["question"], "answer": d["answer"], "evidence": d["evidence"]}
            f.write(json.dumps(row) + "\n")
            kept += 1
            print(i, kept, d["question"], flush=True)
    print("kept", kept)


if __name__ == "__main__":
    main()
