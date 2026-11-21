"""Build the DuckDB warehouse the SQL agent queries.

Tables
- financials:    one row per (ticker, fiscal_year) with annual 10-K figures from SEC XBRL
- online_retail: UCI Online Retail II invoice lines (2009-2011, UK-based online retailer)
"""
from __future__ import annotations

import json
import sys
from datetime import date

import duckdb
import pandas as pd

from . import config

sys.path.insert(0, str(config.ROOT / "scripts"))
from download_data import COMPANIES, fiscal_year  # noqa: E402

# metric -> XBRL concepts in priority order (companies tag the same idea differently)
METRICS = {
    "revenue": ["Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax", "SalesRevenueNet"],
    "cost_of_sales": ["CostOfGoodsAndServicesSold", "CostOfRevenue", "CostOfGoodsSold"],
    "gross_profit": ["GrossProfit"],
    "sga_expense": ["SellingGeneralAndAdministrativeExpense"],
    "operating_income": ["OperatingIncomeLoss"],
    "net_income": ["NetIncomeLoss", "ProfitLoss"],
    "eps_diluted": ["EarningsPerShareDiluted"],
    "total_assets": ["Assets"],
    "inventory": ["InventoryNet"],
    "cash": ["CashAndCashEquivalentsAtCarryingValue"],
    "shareholders_equity": ["StockholdersEquity"],
    "long_term_debt": ["LongTermDebtNoncurrent"],
    "operating_cash_flow": ["NetCashProvidedByUsedInOperatingActivities"],
    "capex": ["PaymentsToAcquirePropertyPlantAndEquipment", "PaymentsToAcquireProductiveAssets"],
    "share_repurchases": ["PaymentsForRepurchaseOfCommonStock"],
    "dividends_paid": ["PaymentsOfDividendsCommonStock", "PaymentsOfDividends"],
}
FIRST_YEAR = 2020


def _annual_facts(concept: dict) -> dict[str, float]:
    """period_end -> value for full-year 10-K facts; the most recent filing wins
    (it carries any restatement)."""
    units = concept["units"]
    unit = max(units, key=lambda u: len(units[u]))  # the main unit (e.g. USD/shares, not pure)
    out: dict[str, tuple[str, float]] = {}
    for f in units[unit]:
        if f.get("form") != "10-K":
            continue
        if "start" in f:  # flow metric: keep full-year durations only
            days = (date.fromisoformat(f["end"]) - date.fromisoformat(f["start"])).days
            if not 350 <= days <= 380:
                continue
        prev = out.get(f["end"])
        if prev is None or f["filed"] >= prev[0]:
            out[f["end"]] = (f["filed"], float(f["val"]))
    return {end: v for end, (_, v) in out.items()}


def financials_frame() -> pd.DataFrame:
    rows: dict[tuple[str, str], dict] = {}
    for ticker, (name, _) in COMPANIES.items():
        gaap = json.loads((config.RAW / f"companyfacts_{ticker}.json").read_text())["facts"]["us-gaap"]
        # fiscal year ends come from a balance-sheet concept every filer reports
        ends = sorted(e for e in _annual_facts(gaap["Assets"]) if fiscal_year(ticker, e) >= FIRST_YEAR)
        for end in ends:
            rows[(ticker, end)] = {"ticker": ticker, "company": name,
                                   "fiscal_year": fiscal_year(ticker, end), "period_end": end}
        for metric, concepts in METRICS.items():
            for c in concepts:
                if c not in gaap:
                    continue
                for end, v in _annual_facts(gaap[c]).items():
                    # flow facts can end a day or two off the balance-sheet date
                    match = next((k for k in rows if k[0] == ticker and
                                  abs((date.fromisoformat(k[1]) - date.fromisoformat(end)).days) <= 7), None)
                    if match and metric not in rows[match]:
                        rows[match][metric] = v
    df = pd.DataFrame(rows.values()).sort_values(["ticker", "fiscal_year"])
    if "gross_profit" in df and "cost_of_sales" in df:  # derive when not tagged directly
        df["gross_profit"] = df["gross_profit"].fillna(df["revenue"] - df["cost_of_sales"])
    return df[["ticker", "company", "fiscal_year", "period_end", *METRICS]]


def online_retail_frame() -> pd.DataFrame:
    cache = config.PROCESSED / "online_retail.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    sheets = pd.read_excel(config.RAW / "online_retail_II.xlsx", sheet_name=None,
                           dtype={"Invoice": str, "StockCode": str, "Customer ID": "Int64"})
    df = pd.concat(sheets.values(), ignore_index=True)
    df.columns = ["invoice", "stock_code", "description", "quantity", "invoice_date",
                  "unit_price", "customer_id", "country"]
    df["description"] = df["description"].astype("string").str.strip()
    df["is_cancellation"] = df["invoice"].str.startswith("C")
    df["line_revenue"] = (df["quantity"] * df["unit_price"]).round(2)
    df.to_parquet(cache)
    return df


SCHEMA_DOC = """\
Table financials -- one row per company fiscal year, from SEC 10-K XBRL. Money in USD (not millions).
  ticker VARCHAR ('WMT','TGT','COST','HD','LOW','BBY'), company VARCHAR,
  fiscal_year INTEGER (the company's own label, e.g. Walmart fiscal 2026 ends Jan 2026),
  period_end DATE, revenue, cost_of_sales, gross_profit, sga_expense, operating_income,
  net_income, eps_diluted (USD per share), total_assets, inventory, cash, shareholders_equity,
  long_term_debt, operating_cash_flow, capex, share_repurchases, dividends_paid  (all DOUBLE, may be NULL)
  revenue is total revenue (includes membership fees for WMT/COST). gross_profit = revenue - cost_of_sales
  when the company does not tag it. Margins: divide by revenue.

Table online_retail -- invoice lines of a UK online gift retailer, Dec 2009 - Dec 2011. Money in GBP.
  invoice VARCHAR (starts with 'C' for cancellations), stock_code VARCHAR, description VARCHAR,
  quantity INTEGER, invoice_date TIMESTAMP, unit_price DOUBLE, customer_id BIGINT (may be NULL),
  country VARCHAR, is_cancellation BOOLEAN, line_revenue DOUBLE (quantity * unit_price)
"""


def build() -> dict[str, int]:
    config.PROCESSED.mkdir(parents=True, exist_ok=True)
    fin, retail = financials_frame(), online_retail_frame()
    con = duckdb.connect(str(config.DB_PATH))
    con.execute("CREATE OR REPLACE TABLE financials AS SELECT * EXCLUDE (period_end), "
                "CAST(period_end AS DATE) AS period_end FROM fin")
    con.execute("CREATE OR REPLACE TABLE online_retail AS SELECT * FROM retail")
    counts = {t: con.execute(f"SELECT count(*) FROM {t}").fetchone()[0] for t in ("financials", "online_retail")}
    con.close()
    return counts


if __name__ == "__main__":
    print(build())
