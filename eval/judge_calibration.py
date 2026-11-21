"""Hand-made true/false cases used to choose the faithfulness judge prompt.  python -m eval.judge_calibration"""
from copilot import llm
J = """You are checking an answer against its sources.
Notes: tickers WMT=Walmart, TGT=Target, COST=Costco, HD=Home Depot, LOW=Lowe's, BBY=Best Buy. An [SQL]
source's WHERE clause tells you which company and fiscal year its result row is for. Rounding
(713,163,000,000 -> 713.16 billion) and simple arithmetic on source values count as supported.

Sources:
{src}

Answer: {ans}

First list each factual claim in the answer and quote the source text that supports it, or write
NONE. Then on the last line reply with JSON only: {{"faithful": true}} if every claim has support,
otherwise {{"faithful": false}}."""
sql = "[SQL] SQL: SELECT revenue FROM financials WHERE ticker='WMT' AND fiscal_year=2026\nResult:\nrevenue\n713,163,000,000 (713.16 billion)"
doc = "[1] Costco FY2026 10-K, Item 1 Business:\nThe number of warehouses with gas stations varies significantly by country. We operated 774 gas stations at the end of 2026."
cases = [(sql, "Walmart's total revenue in fiscal 2026 was 713.16 billion [SQL].", True),
         (sql, "Walmart's total revenue in fiscal 2026 was $713 billion [SQL].", True),
         (sql, "Walmart's total revenue in fiscal 2026 was 650.2 billion [SQL].", False),
         (sql, "Walmart's revenue in fiscal 2026 was 713.16 billion, driven by strong eCommerce growth [SQL].", False),
         (doc, "Costco operated 774 gas stations at the end of fiscal 2026 [1].", True),
         (doc, "Costco operated 774 gas stations at the end of fiscal 2026, mostly in the U.S. [1].", False),
         (doc, "Costco operated 747 gas stations at the end of fiscal 2026 [1].", False),
         (doc + "\n\n" + sql, "Costco operated 774 gas stations [1]; Walmart revenue was 713.16 billion [SQL].", True)]
ok = 0
for src, ans, want in cases:
    out = llm.chat([{"role": "user", "content": J.format(src=src, ans=ans)}], max_tokens=500)
    import re; m = re.findall(r'"faithful"\s*:\s*(true|false)', out.text)
    got = None if not m else m[-1] == "true"
    ok += got == want; print(want, got)
print(ok, "/", len(cases))
