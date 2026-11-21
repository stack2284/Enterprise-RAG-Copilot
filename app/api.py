"""FastAPI service.  uvicorn app.api:app --port 8000

Access control: the X-User header picks a role from USERS; each role sees only its tickers
(both in retrieval and in SQL rows)."""
from __future__ import annotations

import json
import time
import uuid

from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel

from copilot import config, graph

USERS = {  # demo roles; a real deployment would read these from an identity provider
    "analyst": None,                      # all companies
    "home-improvement": ["HD", "LOW"],
    "general-merch": ["WMT", "TGT", "COST", "BBY"],
}

app = FastAPI(title="Enterprise Data Copilot")


class Ask(BaseModel):
    question: str
    history: list[dict] = []
    strategy: str = "semantic"


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "llm": config.LLM_MODEL, "embedder": config.EMBED_MODEL}


@app.post("/ask")
def ask(body: Ask, x_user: str = Header(default="analyst")) -> dict:
    if x_user not in USERS:
        raise HTTPException(403, f"unknown user role {x_user!r}")
    if body.strategy not in ("semantic", "fixed"):
        raise HTTPException(400, "strategy must be 'semantic' or 'fixed'")
    t0, rid = time.perf_counter(), uuid.uuid4().hex[:12]
    st = graph.ask(body.question, history=body.history, allowed_tickers=USERS[x_user], strategy=body.strategy)
    sql = st.get("sql")
    resp = {
        "id": rid,
        "answer": st["answer"],
        "route": st.get("route"),
        "standalone_question": st.get("standalone"),
        "refused": st.get("refused", False),
        "citations": st.get("citations", []),
        "sql": sql.sql if sql else None,
        "sql_rows": [list(map(str, r)) for r in sql.rows[:20]] if sql and sql.ok else None,
        "sql_columns": sql.columns if sql and sql.ok else None,
        "sources": [{k: c[k] for k in ("chunk_id", "company", "fiscal_year", "item", "section", "score", "text")}
                    for c in st.get("chunks") or []],
        "unsupported_numbers": st.get("unsupported_numbers", []),
        "trace": st.get("trace", []),
        "latency_ms": round((time.perf_counter() - t0) * 1000),
        "tokens": sum(t["tokens"] for t in st.get("trace", [])),
    }
    config.LOG_DIR.mkdir(exist_ok=True)
    with open(config.LOG_DIR / "requests.jsonl", "a") as f:  # observability: one line per request
        f.write(json.dumps({"id": rid, "ts": time.time(), "user": x_user, "question": body.question,
                            "route": resp["route"], "refused": resp["refused"], "latency_ms": resp["latency_ms"],
                            "tokens": resp["tokens"], "trace": resp["trace"],
                            "n_unsupported_numbers": len(resp["unsupported_numbers"])}) + "\n")
    return resp
