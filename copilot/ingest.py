"""Parse 10-K HTML into sections, then chunk with a fixed-size or a semantic chunker.

Output: data/processed/chunks_{fixed,semantic}.jsonl, one chunk per line with metadata.
"""
from __future__ import annotations

import json
import re
import warnings
from dataclasses import asdict, dataclass

import numpy as np
from bs4 import BeautifulSoup, NavigableString, XMLParsedAsHTMLWarning

from . import config

warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

ITEM_ORDER = ["1", "1A", "1B", "1C", "2", "3", "4", "5", "6", "7", "7A", "8",
              "9", "9A", "9B", "9C", "10", "11", "12", "13", "14", "15", "16"]
HEADING = re.compile(r"^item\s*(\d{1,2}[a-c]?)\s*[.:\-—–]?\s*(.{3,150})$", re.I)
BLOCK_TAGS = ["p", "div", "br", "tr", "li", "h1", "h2", "h3", "h4", "h5", "h6"]
# Item 8 (financial statements) is mostly numeric tables, which the SQL warehouse covers
# better; items 10-16 are cross-references to the proxy statement and exhibit lists.
SKIP_ITEMS = {"8", "10", "11", "12", "13", "14", "15", "16"}


@dataclass
class Chunk:
    chunk_id: str
    ticker: str
    company: str
    fiscal_year: int
    period_end: str
    item: str
    section: str
    text: str
    source_url: str
    strategy: str


def html_to_lines(raw: bytes) -> list[str]:
    soup = BeautifulSoup(raw, "lxml")
    for t in soup(["script", "style"]):
        t.decompose()
    for t in soup.find_all(lambda x: x.name and x.name.lower() == "ix:header"):
        t.decompose()
    # flatten tables to "a | b | c" rows so numbers stay next to their labels
    for table in soup.find_all("table"):
        rows = []
        for tr in table.find_all("tr"):
            cells = [c.get_text(" ", strip=True) for c in tr.find_all(["td", "th"])]
            cells = [c for c in cells if c and c not in {"$", "%", ")"}]
            if cells:
                rows.append(" | ".join(cells))
        table.replace_with(NavigableString("\n" + "\n".join(rows) + "\n"))
    for tag in soup.find_all(BLOCK_TAGS):
        tag.insert_after(NavigableString("\n"))
    text = soup.get_text("").replace("\xa0", " ")
    lines = [re.sub(r"\s+", " ", ln).strip() for ln in text.split("\n")]
    return [ln for ln in lines if ln and not RUNNING_HEADER.search(ln)]


# page numbers, "Table of Contents" links and running page headers/footers
RUNNING_HEADER = re.compile(
    r"^\d{1,3}$|^table of contents$|form 10-k\s*\|\s*\d+|\|\s*table of contents|^index to financial statements$",
    re.I)


def split_sections(lines: list[str]) -> list[tuple[str, str, list[str]]]:
    """Return (item, title, lines). Headings must appear in increasing item order,
    which skips the table of contents and in-text cross references."""
    sections: list[tuple[str, str, list[str]]] = []
    pos = -1
    for ln in lines:
        m = HEADING.match(ln)
        if m:
            item = m.group(1).upper()
            title = m.group(2).strip(" .—-|")
            is_toc_row = re.search(r"\|\s*\d+\s*$", ln) is not None  # "Item 7 | MD&A | 33"
            if (item in ITEM_ORDER and ITEM_ORDER.index(item) > pos and not is_toc_row
                    and re.search(r"[a-z]{3}", title, re.I)):
                title = title.replace(" | ", " ")
                title = re.sub(r"'S\b", "'s", title.title()) if title.isupper() else title
                pos = ITEM_ORDER.index(item)
                sections.append((item, title, []))
                continue
        if sections:
            sections[-1][2].append(ln)
    return sections


def paragraphs(lines: list[str]) -> list[str]:
    """Merge short heading/table lines with neighbours so every paragraph carries context."""
    paras: list[str] = []
    buf = ""
    for ln in lines:
        buf = f"{buf} {ln}".strip() if buf else ln
        if len(buf.split()) >= 25 or (ln.endswith(".") and len(buf.split()) >= 8):
            paras.append(buf)
            buf = ""
    if buf:
        paras.append(buf)
    return paras


def fixed_chunks(paras: list[str]) -> list[str]:
    words = " ".join(paras).split()
    step = config.CHUNK_WORDS - config.FIXED_OVERLAP_WORDS
    return [" ".join(words[i:i + config.CHUNK_WORDS]) for i in range(0, max(len(words), 1), step)
            if words[i:i + config.CHUNK_WORDS]]


def semantic_chunks(paras: list[str], embs: np.ndarray, threshold: float) -> list[str]:
    """Pack whole paragraphs; start a new chunk at a topic shift (low cosine similarity
    between neighbouring paragraphs) or when the size budget is reached."""
    chunks, cur, cur_len = [], [], 0
    for i, p in enumerate(paras):
        n = len(p.split())
        if n > config.CHUNK_WORDS * 1.5:  # very long paragraph: split it by sentences
            if cur:
                chunks.append(" ".join(cur)); cur, cur_len = [], 0
            sents = re.split(r"(?<=[.;])\s+", p)
            buf: list[str] = []
            for s in sents:
                if buf and len(" ".join(buf + [s]).split()) > config.CHUNK_WORDS:
                    chunks.append(" ".join(buf)); buf = []
                buf.append(s)
            if buf:
                chunks.append(" ".join(buf))
            continue
        topic_shift = i > 0 and float(embs[i] @ embs[i - 1]) < threshold and cur_len >= 60
        if cur and (cur_len + n > config.CHUNK_WORDS or topic_shift):
            chunks.append(" ".join(cur)); cur, cur_len = [], 0
        cur.append(p); cur_len += n
    if cur:
        chunks.append(" ".join(cur))
    return chunks


def build() -> dict[str, int]:
    from sentence_transformers import SentenceTransformer

    manifest = json.loads((config.FILINGS / "manifest.json").read_text())
    model = SentenceTransformer(config.EMBED_MODEL)
    config.PROCESSED.mkdir(parents=True, exist_ok=True)
    out = {s: open(config.PROCESSED / f"chunks_{s}.jsonl", "w") for s in ("fixed", "semantic")}
    counts = {"fixed": 0, "semantic": 0, "sections": 0}
    for doc in manifest:
        lines = html_to_lines((config.FILINGS / doc["file"]).read_bytes())
        for item, title, sec_lines in split_sections(lines):
            if item in SKIP_ITEMS:
                continue
            paras = paragraphs(sec_lines)
            if not paras:
                continue
            counts["sections"] += 1
            embs = model.encode(paras, normalize_embeddings=True, batch_size=64)
            sims = (embs[1:] * embs[:-1]).sum(1) if len(paras) > 1 else np.array([1.0])
            threshold = float(np.percentile(sims, config.SEMANTIC_BREAK_PCTL))
            variants = {"fixed": fixed_chunks(paras), "semantic": semantic_chunks(paras, embs, threshold)}
            for strategy, texts in variants.items():
                for j, t in enumerate(texts):
                    c = Chunk(
                        chunk_id=f"{doc['ticker']}-{doc['fiscal_year']}-{item}-{strategy[0]}{j:03d}",
                        ticker=doc["ticker"], company=doc["company"], fiscal_year=doc["fiscal_year"],
                        period_end=doc["period_end"], item=item, section=title, text=t,
                        source_url=doc["url"], strategy=strategy,
                    )
                    out[strategy].write(json.dumps(asdict(c)) + "\n")
                    counts[strategy] += 1
        print(f"parsed {doc['file']}", flush=True)
    for f in out.values():
        f.close()
    return counts


if __name__ == "__main__":
    print(build())
