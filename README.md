# Enterprise Data Copilot

An agentic RAG assistant that answers questions over **unstructured** text (12 SEC 10-K filings from
Walmart, Target, Costco, Home Depot, Lowe's and Best Buy, two fiscal years each) and a **structured
warehouse** (DuckDB: XBRL financials for the same companies plus 1.07M rows of UCI Online Retail II
transactions). A LangGraph agent decides per question whether to search documents, write SQL, or both.
It cites every claim, checks that its numbers come from the sources, and refuses when the data doesn't
cover the question. Everything runs locally on an Apple-silicon Mac; no API keys are needed.

```
question ─► rewrite (chat memory → standalone question)
         ─► route (LLM: doc | sql | hybrid | none)
               ├─ doc ──► hybrid retrieval: metadata filters → FAISS HNSW + BM25 → RRF → cross-encoder rerank
               ├─ sql ──► text-to-SQL → read-only validation → DuckDB → retry with the error on failure
               └─ hybrid ► both
         ─► synthesize (cited answer, or an exact refusal) ─► numeric grounding check
```

## Stack

| Layer | Choice |
|---|---|
| LLM | Qwen2.5-7B-Instruct (4-bit) through MLX, temperature 0 |
| Embeddings / rerank | BAAI/bge-small-en-v1.5, cross-encoder/ms-marco-MiniLM-L-6-v2 |
| Vector + lexical search | FAISS HNSW (M=32), rank_bm25, reciprocal rank fusion |
| Warehouse | DuckDB (`financials` wide table from XBRL, `online_retail` transactions) |
| Orchestration | LangGraph `StateGraph` |
| Serving | FastAPI (`/ask`, `/health`, per-request JSONL logs), Streamlit chat UI |

## Design notes

- **Two chunkers.** Fixed 220-word windows with 40-word overlap, and semantic chunks that split where the
  similarity between neighbouring paragraphs falls in the bottom 20%. Tables are flattened to `a | b` rows
  and each chunk gets a header naming the company, fiscal year and 10-K item.
- **Metadata filters.** The question is parsed for company aliases and fiscal years and the search is
  restricted to matching chunks. If nothing matches, the filter falls back to company only.
- **Access control.** The `X-User` header maps to a role whose allowed tickers mask both retrieval and
  SQL result rows (`analyst` sees all; `home-improvement` sees HD and LOW; `general-merch` sees the rest).
- **SQL safety.** Only single `SELECT`/`WITH` statements pass validation, a keyword denylist blocks
  writes, and the connection itself is read-only.
- **Grounding.** Every sentence must cite `[n]` or `[SQL]`. After generation, each number in the answer
  is checked against the sources, and any untraceable number is flagged in the response and the UI.
- **Fiscal years.** Labels follow each company's own convention. For example, Best Buy's year ending
  January 2026 is "fiscal 2026", while Target's year ending January 2026 is "fiscal 2025".

## Evaluation

The golden set has 118 questions (`eval/golden.jsonl`):

- **68 document questions.** Qwen drafted them from sampled 10-K paragraphs. Each was then reviewed by
  hand and kept only if it named the company and year and its answer matched a verbatim evidence sentence.
- **25 SQL questions** with reference SQL.
- **15 hybrid questions** (a number plus an explanation).
- **10 unanswerable questions**, such as other companies, stock prices or future years.

### Retrieval (68 doc questions, k=10)

A hit means a retrieved chunk from the right company and year contains the evidence sentence.

| Chunking | Retriever | Filters | Recall@1 | Recall@5 | MRR@10 | p50 latency |
|---|---|---|---|---|---|---|
| fixed | dense (baseline) | no | 0.529 | 0.809 | 0.644 | 6.5 ms |
| fixed | dense | yes | 0.574 | 0.824 | 0.685 | 6.4 ms |
| fixed | BM25 | yes | 0.721 | 0.926 | 0.817 | 1.2 ms |
| fixed | hybrid + rerank | yes | 0.779 | 0.926 | 0.848 | 129 ms |
| semantic | dense | yes | 0.647 | 0.794 | 0.717 | 7.0 ms |
| semantic | BM25 | yes | 0.824 | 0.941 | 0.883 | 1.7 ms |
| semantic | hybrid (RRF) | yes | 0.735 | 0.882 | 0.804 | 9.4 ms |
| semantic | hybrid + rerank | no | 0.750 | 0.926 | 0.836 | 126 ms |
| **semantic** | **hybrid + rerank** | **yes** | **0.853** | 0.912 | 0.880 | 111 ms |

The full 16-row grid is in `eval/results/retrieval.json`. Semantic chunking, metadata filters and
reranking each help, and together they lift Recall@1 from 53% to 85%. BM25 alone is almost as strong
here. That is likely because the questions were drafted from the passages they target, so they share
vocabulary with them; paraphrased user questions would probably favour dense retrieval more.

### Text-to-SQL (25 questions)

The score is execution accuracy: the reference query's values must appear in the agent's result.

| Execution accuracy | First attempt | Recovered by retry | Valid SQL |
|---|---|---|---|
| **84%** (21/25) | 76% | 2 | 100% |

The remaining misses are counting rows instead of distinct invoices, grouping by month without the
year, averaging line values instead of order values, and one misread question. The prompt was not tuned
against these questions.

### End to end (118 questions): agent vs naive RAG

Results are from `eval/results/e2e_*_v2.json`.

| Metric | Naive RAG | Agent |
|---|---|---|
| Answer accuracy, 108 answerable questions | 80.6% | **88.9%** |
| doc (68) / sql (25) / hybrid (15) correct | 88% / 76% / 53% | 85% / **100%** / **87%** |
| Faithful answers (LLM judge, answered questions) | 53.5% | **84.3%** |
| Unanswerable questions refused (10) | 0/10 | **10/10** |
| False refusals on answerable questions | 0% | 3.7% |
| Answers with a number not found in the sources | 2.5% | **0%** |
| p50 / p95 latency | 7.3 s / 14.1 s | 7.6 s / 14.8 s |
| Mean tokens per question | 2,769 | 3,052 |

The agent cuts unsupported answers from 46% to 16% and refuses every out-of-scope question. On plain
document questions the baseline is slightly ahead (88% vs 85%), because the agent sometimes pulls in an
irrelevant SQL figure or refuses.

**How the agent got here (v1 → v2).** The first full run (`*_v1.json`) trusted the LLM router outright.
The router misrouted 42% of questions, and 27% of answerable questions were refused or sent to SQL
alone, so doc accuracy was only 44%. The fix makes the route advisory:

- Refuse up front only when the question names none of the six retailers or the retail dataset.
- Always search the 10-K text, even when SQL runs too.

The same run showed that the faithfulness judge rejected even verbatim-supported answers. It was
replaced with a quote-then-verdict prompt, chosen on 8 hand-made cases (`eval/judge_calibration.py`,
7/8 correct). Both systems were then re-scored with the new judge. These changes came from error
analysis on this same golden set, so the v2 numbers are optimistic. A held-out set is the obvious next step.

The naive RAG baseline uses dense top-8 retrieval over fixed chunks with a plain prompt; it has no
filters, rerank, SQL, router or refusal instruction. Doc-question correctness and faithfulness are
scored by the same local Qwen model acting as judge. SQL and hybrid correctness require the reference
figure to appear in the answer.

### Limitations

- The doc questions were LLM-drafted from the passages they target, so retrieval numbers are optimistic.
- The judge is the same 7B model that generates the answers. It is useful for comparing systems, but it
  is not a ground-truth measure.
- After a five-question smoke test (separate from the golden set), the context size was raised from 6 to
  8 chunks and the router prompt was given an explicit rule for two-part questions.
- The LLM router's own label matches the expected route only 58% of the time; the agent no longer
  depends on it except for out-of-scope refusals.
- Some "misses" are scorer strictness. For example, Costco's net sales ($269.9B) is marked wrong against
  total revenue including membership fees ($275.2B).

## Run it

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
SEC_CONTACT="you@example.com" .venv/bin/python scripts/download_data.py   # SEC asks for a contact in the UA
.venv/bin/python -m copilot.ingest        # parse + chunk 10-Ks
.venv/bin/python -m copilot.structured    # build DuckDB warehouse
.venv/bin/uvicorn app.api:app --port 8000
.venv/bin/streamlit run app/ui.py
```

Evaluate:

```bash
.venv/bin/python -m eval.run_eval retrieval
.venv/bin/python -m eval.run_eval sql
.venv/bin/python -m eval.run_eval e2e
```

Data: SEC EDGAR filings and XBRL company facts (public), and the
[UCI Online Retail II](https://archive.ics.uci.edu/dataset/502/online+retail+ii) dataset (CC BY 4.0).
