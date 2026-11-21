"""Text-to-SQL agent over the DuckDB warehouse: generate, validate (read-only), execute,
and on a database error feed the error back to the model and retry."""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import duckdb

from . import config, llm
from .structured import SCHEMA_DOC

MAX_ATTEMPTS = 3
ROW_LIMIT = 50
FORBIDDEN = re.compile(r"\b(insert|update|delete|drop|create|alter|attach|detach|copy|export|import|"
                       r"install|load|pragma|set|call|replace)\b", re.I)

SYSTEM = f"""You translate questions into one DuckDB SQL query.
{SCHEMA_DOC}
Rules: write a single SELECT (or WITH ... SELECT). Use only the columns above. Return the columns that
answer the question (include ticker/fiscal_year when comparing companies or years). For ratios and
growth, compute them in SQL. Exclude cancellations for online retail sales unless asked.
Reply with JSON only: {{"sql": "..."}}

Examples:
Q: What was Best Buy's operating income in fiscal 2023?
{{"sql": "SELECT ticker, fiscal_year, operating_income FROM financials WHERE ticker='BBY' AND fiscal_year=2023"}}
Q: Which company had the most cash at the end of fiscal 2024?
{{"sql": "SELECT ticker, cash FROM financials WHERE fiscal_year=2024 ORDER BY cash DESC LIMIT 1"}}
Q: In the online retail data, how many units of products were sold to Germany in 2010?
{{"sql": "SELECT sum(quantity) AS units FROM online_retail WHERE country='Germany' AND NOT is_cancellation AND year(invoice_date)=2010"}}
"""


@dataclass
class SQLResult:
    sql: str | None = None
    columns: list[str] = field(default_factory=list)
    rows: list[tuple] = field(default_factory=list)
    error: str | None = None
    attempts: int = 0
    tokens: int = 0

    @property
    def ok(self) -> bool:
        return self.error is None and self.sql is not None


def validate(sql: str) -> str | None:
    s = sql.strip().rstrip(";")
    if ";" in s:
        return "only one statement is allowed"
    if not re.match(r"^\s*(select|with)\b", s, re.I):
        return "query must start with SELECT or WITH"
    if FORBIDDEN.search(re.sub(r"'[^']*'", "''", s)):  # ignore words inside string literals
        return "query must be read-only"
    return None


def connect() -> duckdb.DuckDBPyConnection:
    return duckdb.connect(str(config.DB_PATH), read_only=True)


def run(question: str) -> SQLResult:
    res = SQLResult()
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": f"Q: {question}"}]
    con = connect()
    try:
        for attempt in range(1, MAX_ATTEMPTS + 1):
            res.attempts = attempt
            data, out = llm.chat_json(messages, max_tokens=300)
            res.tokens += out.prompt_tokens + out.completion_tokens
            sql = (data or {}).get("sql", "").strip().rstrip(";")
            err = "reply was not JSON with a 'sql' field" if not sql else validate(sql)
            if err is None:
                try:
                    cur = con.execute(sql)
                    res.columns = [d[0] for d in cur.description]
                    res.rows = cur.fetchmany(ROW_LIMIT)
                    res.sql, res.error = sql, None
                    return res
                except duckdb.Error as e:
                    err = str(e).split("\n")[0][:300]
            res.sql, res.error = sql or None, err
            messages += [{"role": "assistant", "content": out.text},
                         {"role": "user", "content": f"That failed: {err}\nFix it. Reply with JSON only."}]
        return res
    finally:
        con.close()


def humanize(v) -> str:
    if isinstance(v, float) or isinstance(v, int) and not isinstance(v, bool):
        x = float(v)
        if abs(x) >= 1e9:
            return f"{x:,.0f} ({x/1e9:,.2f} billion)"
        if abs(x) >= 1e6:
            return f"{x:,.0f} ({x/1e6:,.2f} million)"
        if abs(x) < 1 and x != 0:
            return f"{x:.4f} ({x*100:.2f}%)"
        return f"{x:,.2f}" if isinstance(v, float) else f"{v:,}"
    return str(v)


def as_text(r: SQLResult) -> str:
    if not r.ok:
        return f"SQL failed: {r.error}"
    if not r.rows:
        return f"SQL: {r.sql}\nResult: no rows"
    lines = [" | ".join(r.columns)] + [" | ".join(humanize(v) for v in row) for row in r.rows]
    return f"SQL: {r.sql}\nResult:\n" + "\n".join(lines)
