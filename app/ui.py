"""Streamlit chat UI.  streamlit run app/ui.py   (expects the API on localhost:8000)"""
from __future__ import annotations

import os

import requests
import streamlit as st

API = os.environ.get("COPILOT_API", "http://localhost:8000")

st.set_page_config(page_title="Enterprise Data Copilot", layout="wide")
st.title("Enterprise Data Copilot")
st.caption("Ask about Walmart, Target, Costco, Home Depot, Lowe's and Best Buy 10-Ks, their financials, "
           "or the Online Retail II sales data.")

with st.sidebar:
    user = st.selectbox("User role (access control)", ["analyst", "home-improvement", "general-merch"])
    strategy = st.selectbox("Chunking", ["semantic", "fixed"])
    if st.button("Clear chat"):
        st.session_state.messages = []

st.session_state.setdefault("messages", [])
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

if q := st.chat_input("e.g. What drove Target's comparable sales change in fiscal 2025?"):
    st.chat_message("user").markdown(q)
    history = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]
    with st.chat_message("assistant"), st.spinner("Thinking..."):
        r = requests.post(f"{API}/ask", json={"question": q, "history": history, "strategy": strategy},
                          headers={"X-User": user}, timeout=300)
        r.raise_for_status()
        d = r.json()
        st.markdown(d["answer"])
        st.caption(f"route: {d['route']} | {d['latency_ms']} ms | {d['tokens']} tokens"
                   + (f" | rewritten: {d['standalone_question']}" if d["standalone_question"] != q else ""))
        if d["unsupported_numbers"]:
            st.warning(f"Numbers not found in sources: {', '.join(d['unsupported_numbers'])}")
        if d["sql"]:
            with st.expander("SQL"):
                st.code(d["sql"], language="sql")
                if d["sql_rows"]:
                    st.table([dict(zip(d["sql_columns"], row)) for row in d["sql_rows"]])
        if d["sources"]:
            with st.expander(f"Sources ({len(d['sources'])})"):
                for i, s in enumerate(d["sources"], 1):
                    st.markdown(f"**[{i}] {s['company']} FY{s['fiscal_year']}, Item {s['item']} {s['section']}** "
                                f"(rerank {s['score']:.2f})")
                    st.write(s["text"])
    st.session_state.messages += [{"role": "user", "content": q}, {"role": "assistant", "content": d["answer"]}]
