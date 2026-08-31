import hashlib
import json
import re
import sqlite3
from typing import Callable, Optional

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer

# src/ klasörü altından içe aktarım
from src.config import (
    BM25_CANDIDATES,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    DB_PATH,
    DENSE_CANDIDATES,
    EMBEDDING_MODEL,
    RERANK_CANDIDATES,
    RERANKER_MODEL,
    RETRIEVAL_LIMIT,
)

_embedder: SentenceTransformer | None = None
_reranker: CrossEncoder | None = None


def embedding_model() -> SentenceTransformer:
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(EMBEDDING_MODEL, device="cpu")
    return _embedder


def reranker_model() -> CrossEncoder:
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoder(RERANKER_MODEL, device="cpu")
    return _reranker


def tokens(text: str) -> list[str]:
    return re.findall(r"[a-zA-ZçÇğĞıİöÖşŞüÜ0-9_-]+", (text or "").lower())


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").replace("\x00", " ")).strip()


def chunks_from_sections(sections: list[dict]) -> list[dict]:
    chunks = []
    for section_no, section in enumerate(sections, start=1):
        text = normalize(str(section.get("text", "")))
        location = str(section.get("location") or f"Bölüm {section_no}")
        if not text:
            continue
        start, number = 0, 1
        while start < len(text):
            end = min(start + CHUNK_SIZE, len(text))
            part = text[start:end]
            if end < len(text):
                cut = max(part.rfind(". "), part.rfind("! "), part.rfind("? "))
                if cut > CHUNK_SIZE // 2:
                    end = start + cut + 1
                    part = text[start:end]
            part = normalize(part)
            if part:
                chunks.append({"location": location, "number": number, "text": part})
            if end >= len(text):
                break
            start = max(start + 1, end - CHUNK_OVERLAP)
            number += 1
    return chunks


def connection() -> sqlite3.Connection:
    db = sqlite3.connect(DB_PATH)
    db.row_factory = sqlite3.Row
    db.execute("PRAGMA journal_mode=WAL")
    db.execute("CREATE TABLE IF NOT EXISTS documents (document_id TEXT PRIMARY KEY, filename TEXT NOT NULL, file_hash TEXT NOT NULL, metadata TEXT NOT NULL, created_at TEXT DEFAULT CURRENT_TIMESTAMP)")
    db.execute("CREATE TABLE IF NOT EXISTS chunks (chunk_id TEXT PRIMARY KEY, document_id TEXT NOT NULL, location TEXT NOT NULL, chunk_number INTEGER NOT NULL, text TEXT NOT NULL, embedding TEXT NOT NULL)")
    db.commit()
    return db


def index_document(filename: str, data: bytes, sections: list[dict], metadata: dict, progress_callback: Optional[Callable[[int, str], None]] = None) -> dict:
    file_hash = hashlib.sha256(data).hexdigest()
    document_id = hashlib.sha256(f"{filename}:{file_hash}".encode()).hexdigest()
    chunks = chunks_from_sections(sections)
    if not chunks:
        raise ValueError("İndekslenecek metin parçası bulunamadı.")
    db = connection()
    if db.execute("SELECT 1 FROM documents WHERE document_id=?", (document_id,)).fetchone():
        db.close()
        return {"already_indexed": True, "chunk_count": len(chunks)}
    if progress_callback:
        progress_callback(12, f"{len(chunks)} parça hazırlandı. Embedding oluşturuluyor...")
    vectors = []
    batch_size = 24
    for start in range(0, len(chunks), batch_size):
        batch = [chunk["text"] for chunk in chunks[start:start + batch_size]]
        vectors.extend(embedding_model().encode(batch, batch_size=batch_size, normalize_embeddings=True, show_progress_bar=False))
        if progress_callback:
            done = min(start + len(batch), len(chunks))
            progress_callback(15 + int(75 * done / len(chunks)), f"Embedding: %{int(100 * done / len(chunks))} ({done}/{len(chunks)})")
    db.execute("INSERT INTO documents VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)", (document_id, filename, file_hash, json.dumps(metadata, ensure_ascii=False)))
    for index, (chunk, vector) in enumerate(zip(chunks, vectors)):
        chunk_id = hashlib.sha256(f"{document_id}:{index}".encode()).hexdigest()
        db.execute("INSERT INTO chunks VALUES (?, ?, ?, ?, ?, ?)", (chunk_id, document_id, chunk["location"], chunk["number"], chunk["text"], json.dumps(vector.tolist())))
    db.commit()
    db.close()
    if progress_callback:
        progress_callback(100, "İndeks kaydedildi.")
    return {"already_indexed": False, "chunk_count": len(chunks)}


def list_documents() -> list[dict]:
    db = connection()
    rows = db.execute("SELECT d.document_id, d.filename, COUNT(c.chunk_id) AS chunk_count FROM documents d LEFT JOIN chunks c ON d.document_id=c.document_id GROUP BY d.document_id ORDER BY d.created_at DESC").fetchall()
    db.close()
    return [dict(row) for row in rows]


def delete_document(document_id: str) -> None:
    db = connection()
    db.execute("DELETE FROM chunks WHERE document_id = ?", (document_id,))
    db.execute("DELETE FROM documents WHERE document_id = ?", (document_id,))
    db.commit()
    db.close()


def clear_documents() -> None:
    db = connection()
    db.execute("DELETE FROM chunks")
    db.execute("DELETE FROM documents")
    db.commit()
    db.close()


def rrf(rankings: list[list[int]], k: int = 60) -> dict[int, float]:
    result: dict[int, float] = {}
    for ranking in rankings:
        for rank, index in enumerate(ranking, start=1):
            result[index] = result.get(index, 0) + 1 / (k + rank)
    return result


def overlap(a: str, b: str) -> float:
    a_tokens, b_tokens = set(tokens(a)), set(tokens(b))
    return len(a_tokens & b_tokens) / max(1, len(a_tokens | b_tokens))


def retrieve(question: str, limit: int = RETRIEVAL_LIMIT) -> list[dict]:
    db = connection()
    rows = db.execute("SELECT d.filename, c.location, c.chunk_number, c.text, c.embedding FROM chunks c JOIN documents d ON c.document_id=d.document_id").fetchall()
    db.close()
    if not rows:
        return []
    texts = [row["text"] for row in rows]
    matrix = np.asarray([json.loads(row["embedding"]) for row in rows], dtype=np.float32)
    query_vector = embedding_model().encode([question], normalize_embeddings=True, show_progress_bar=False)[0]
    dense = np.dot(matrix, query_vector)
    bm25 = np.asarray(BM25Okapi([tokens(text) for text in texts]).get_scores(tokens(question)), dtype=np.float32)
    dense_rank = list(np.argsort(-dense)[:min(DENSE_CANDIDATES, len(rows))])
    bm25_rank = list(np.argsort(-bm25)[:min(BM25_CANDIDATES, len(rows))])
    fused = rrf([dense_rank, bm25_rank])
    candidates = sorted(fused, key=fused.get, reverse=True)[:RERANK_CANDIDATES]
    scores = reranker_model().predict([(question, texts[i]) for i in candidates], show_progress_bar=False)
    score_by_index = {index: float(score) for index, score in zip(candidates, scores)}
    ordered = sorted(candidates, key=lambda index: score_by_index[index], reverse=True)
    selected = []
    for index in ordered:
        if all(overlap(texts[index], item["text"]) < 0.55 for item in selected):
            selected.append({
                "filename": rows[index]["filename"],
                "location": rows[index]["location"],
                "text": texts[index],
                "semantic_score": round(float(dense[index]), 3),
                "bm25_score": round(float(bm25[index]), 3),
                "rerank_score": round(score_by_index[index], 3),
            })
        if len(selected) >= limit:
            break
    return selected