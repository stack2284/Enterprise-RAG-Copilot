"""Assemble eval/golden.jsonl from manually reviewed doc drafts + hand-written SQL, hybrid and
unanswerable questions.

Review notes: drafts were dropped when the question was vague, circular ("which index ... S&P
500 Index?"), about the document itself (page numbers, URLs, filing dates), or when the drafted
answer was wrong. Kept drafts were edited so every question names the company and fiscal year
and every answer is correct against its evidence sentence.
"""
from __future__ import annotations

import json

from copilot import config
from eval.sql_questions import HYBRID_QUESTIONS, SQL_QUESTIONS, UNANSWERABLE

# id -> {} (keep as drafted) or overrides for question / answer
KEEP = {
    "doc-000": {"answer": "It decreased by $1.3 billion"},
    "doc-001": {}, "doc-002": {}, "doc-003": {}, "doc-005": {}, "doc-006": {},
    "doc-007": {"answer": "A $0.7 billion charge related to PhonePe's modification of share-based awards"},
    "doc-008": {},
    "doc-010": {"question": "According to Walmart's fiscal 2025 10-K, how long has its CISO been a Walmart associate?"},
    "doc-011": {"question": "About how many customers did Walmart serve each week, per its fiscal 2025 10-K?"},
    "doc-012": {"question": "By how much did the net fair value of Walmart's interest rate swaps change in fiscal 2025?",
                "answer": "It increased $43 million"},
    "doc-013": {}, "doc-014": {},
    "doc-016": {"answer": "Net losses of $0.8 billion"},
    "doc-018": {}, "doc-019": {},
    "doc-020": {"question": "According to Target's fiscal 2025 10-K, which country is the single largest source of the merchandise Target imports?"},
    "doc-021": {"question": "According to Target's fiscal 2025 10-K, on what date did the U.S. Supreme Court rule that IEEPA tariffs were not authorized?"},
    "doc-022": {}, "doc-023": {}, "doc-024": {}, "doc-025": {}, "doc-027": {},
    "doc-029": {"question": "Per Target's fiscal 2024 10-K, what position did Cara Sylvester hold before becoming Chief Guest Experience Officer in May 2022?"},
    "doc-030": {}, "doc-031": {}, "doc-033": {},
    "doc-035": {"question": "Per Target's fiscal 2024 10-K, what role has Amy Tu held since August 2024?"},
    "doc-036": {}, "doc-038": {"answer": "$6,175 million"}, "doc-039": {}, "doc-040": {}, "doc-042": {},
    "doc-043": {}, "doc-044": {}, "doc-049": {}, "doc-050": {}, "doc-052": {}, "doc-053": {},
    "doc-057": {"question": "Did Home Depot experience any material cybersecurity incidents in fiscal 2025, per its 10-K?"},
    "doc-058": {},
    "doc-059": {"question": "In what month of fiscal 2025 did Home Depot terminate its three back-up credit facility agreements?"},
    "doc-060": {},
    "doc-062": {"question": "Per Home Depot's fiscal 2025 10-K, how many locations did SRS (including GMS) operate at year end?",
                "answer": "Over 1,250"},
    "doc-063": {"answer": "1.8%"},
    "doc-065": {}, "doc-066": {}, "doc-067": {},
    "doc-069": {"question": "What did Home Depot use to repay about $10.0 billion of commercial paper in June 2024 (fiscal 2024)?",
                "answer": "Proceeds from issuing $10.0 billion of long-term debt"},
    "doc-070": {"question": "Why did Home Depot pause share repurchases in March 2024, per its fiscal 2024 10-K?"},
    "doc-071": {},
    "doc-078": {"question": "When did Lowe's update its Total Home strategy, per its fiscal 2025 10-K?", "answer": "December 2024"},
    "doc-079": {}, "doc-080": {},
    "doc-083": {"question": "Per Lowe's fiscal 2024 10-K, at how many store and distribution center locations did Lowe's plan to install rooftop solar panels?",
                "answer": "174"},
    "doc-084": {"answer": "About $138 million"}, "doc-085": {}, "doc-089": {"answer": "3.1%"},
    "doc-090": {}, "doc-091": {}, "doc-092": {},
    "doc-094": {"question": "Per Best Buy's fiscal 2026 10-K, what instruments does it use to manage foreign currency exposure on forecasted inventory purchases?"},
    "doc-095": {"question": "Per Best Buy's fiscal 2026 10-K, exiting which business led to impairment charges?"},
    "doc-096": {},
    "doc-097": {"question": "Per Best Buy's fiscal 2026 10-K, what share of its overall assortment does Best Buy import directly?"},
    "doc-100": {"question": "Per Best Buy's fiscal 2025 10-K, how many full months must a location operate before its sales count in comparable sales?"},
    "doc-103": {"question": "Per Best Buy's fiscal 2025 10-K, in which state was Best Buy incorporated?"},
    "doc-107": {},
}
BBY_RELABEL = 1  # drafts were made before Best Buy's fiscal-year labels were corrected


def main() -> None:
    drafts = {d["id"]: d for d in map(json.loads, open(config.ROOT / "eval" / "drafts_doc.jsonl"))}
    rows = []
    for qid, over in KEEP.items():
        d = dict(drafts[qid])
        if d["ticker"] == "BBY":
            d["fiscal_year"] += BBY_RELABEL
        d.update(over)
        rows.append(d)
    rows += [{"id": f"sql-{i:03d}", "type": "sql", "question": q, "gold_sql": s}
             for i, (q, s) in enumerate(SQL_QUESTIONS)]
    rows += [{"id": f"hyb-{i:03d}", "type": "hybrid", "question": q, "gold_sql": s, "ticker": t,
              "fiscal_year": fy, "item": item} for i, (q, s, t, fy, item) in enumerate(HYBRID_QUESTIONS)]
    rows += [{"id": f"una-{i:03d}", "type": "unanswerable", "question": q} for i, q in enumerate(UNANSWERABLE)]
    with open(config.ROOT / "eval" / "golden.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    counts = {t: sum(r["type"] == t for r in rows) for t in ("doc", "sql", "hybrid", "unanswerable")}
    print(counts, "total", len(rows))


if __name__ == "__main__":
    main()
