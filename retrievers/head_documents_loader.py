import contextlib
from typing import List, Dict, Iterable, DefaultDict
from collections import defaultdict
from langchain.schema import Document

def _iter_docstore_docs(docstore) -> Iterable[Document]:
    if isinstance(docstore, dict):
        yield from docstore.values(); return
    d = getattr(docstore, "_dict", None)
    if isinstance(d, dict):
        yield from d.values(); return
    with contextlib.suppress(Exception):
        yield from docstore

def _build_heads_by_source(docstore) -> Dict[str, List[Document]]:
    by_src: DefaultDict[str, List[Document]] = defaultdict(list)
    for doc in _iter_docstore_docs(docstore):
        md = (doc.metadata or {})
        if md.get("type", "").lower() == "head" and md.get("source"):
            by_src[md["source"]].append(doc)
    return by_src

def _dedupe_docs(docs: List[Document]) -> List[Document]:
    seen = set()
    out = []
    head_docs=[d for d in docs if d.metadata["type"] == "head"]
    other_docs=[d for d in docs if d.metadata["type"] != "head"]
    for d in head_docs:
        m = d.metadata or {}
        # prefer a stable id if you have one
        key = m.get("id") or m.get("doc_id") or (m.get("source"), m.get("type"), d.page_content)
        if key not in seen:
            seen.add(key)
            out.append(d)
    return out + other_docs