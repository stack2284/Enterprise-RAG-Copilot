"""Download the public datasets used by the copilot.

- 10-K annual reports (HTML) for six US retailers from SEC EDGAR, last 2 fiscal years each
- XBRL "company facts" (structured financials) for the same companies from SEC EDGAR
- UCI Online Retail II (transaction-level sales data)

SEC's fair-access policy asks automated clients to declare a contact email in the
User-Agent. Set it with:  SEC_CONTACT="you@example.com" python scripts/download_data.py
"""
from __future__ import annotations

import io
import json
import os
import time
import zipfile
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
UA = {"User-Agent": f"EnterpriseRAGCopilot/0.1 {os.environ.get('SEC_CONTACT', 'educational-research-script')}"}

COMPANIES = {
    "WMT": ("Walmart", 104169),
    "TGT": ("Target", 27419),
    "COST": ("Costco", 909832),
    "HD": ("Home Depot", 354950),
    "LOW": ("Lowe's", 60667),
    "BBY": ("Best Buy", 764478),
}
TENKS_PER_COMPANY = 2


def get(url: str) -> requests.Response:
    for attempt in range(4):
        r = requests.get(url, headers=UA, timeout=60)
        if r.status_code == 200:
            time.sleep(0.25)  # stay well under SEC's 10 req/s limit
            return r
        time.sleep(2 ** attempt)
    r.raise_for_status()
    return r


def fiscal_year(ticker: str, period_end: str) -> int:
    """Each retailer's own fiscal-year label. Walmart and Best Buy name the year after the
    January it ends in; Target, Home Depot and Lowe's after the calendar year it mostly
    covers; Costco's year ends in late August."""
    y, mo = int(period_end[:4]), int(period_end[5:7])
    return y if (ticker in ("WMT", "COST", "BBY") or mo > 2) else y - 1


def download_filings() -> None:
    out = RAW / "filings"
    out.mkdir(parents=True, exist_ok=True)
    manifest = []
    for ticker, (name, cik) in COMPANIES.items():
        sub = get(f"https://data.sec.gov/submissions/CIK{cik:010d}.json").json()
        recent = sub["filings"]["recent"]
        n = 0
        for form, acc, doc, fdate, rdate in zip(
            recent["form"], recent["accessionNumber"], recent["primaryDocument"],
            recent["filingDate"], recent["reportDate"],
        ):
            if form != "10-K":
                continue
            fy = fiscal_year(ticker, rdate)
            path = out / f"{ticker}_10K_FY{fy}_{rdate}.html"
            url = f"https://www.sec.gov/Archives/edgar/data/{cik}/{acc.replace('-', '')}/{doc}"
            if not path.exists():
                path.write_bytes(get(url).content)
            manifest.append({"ticker": ticker, "company": name, "cik": cik, "form": "10-K",
                             "fiscal_year": fy, "period_end": rdate, "filed": fdate,
                             "url": url, "file": path.name})
            print(f"{ticker} FY{fy}: {path.stat().st_size/1e6:.1f} MB")
            n += 1
            if n == TENKS_PER_COMPANY:
                break
        facts = RAW / f"companyfacts_{ticker}.json"
        if not facts.exists():
            facts.write_bytes(get(f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json").content)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))


def download_online_retail() -> None:
    target = RAW / "online_retail_II.xlsx"
    if target.exists():
        return
    r = get("https://archive.ics.uci.edu/static/public/502/online+retail+ii.zip")
    with zipfile.ZipFile(io.BytesIO(r.content)) as z:
        name = next(n for n in z.namelist() if n.endswith(".xlsx"))
        target.write_bytes(z.read(name))
    print(f"online retail: {target.stat().st_size/1e6:.1f} MB")


if __name__ == "__main__":
    download_filings()
    download_online_retail()
