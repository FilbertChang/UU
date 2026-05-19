"""Vector retrieval over the chunk store: cosine search + optional reranking.

Flow: embed the query -> pgvector cosine nearest-neighbour search (with optional
metadata filters) -> if enabled, rerank the candidates with a cross-encoder and
keep the best `top_k`.
"""

from dataclasses import dataclass
from functools import lru_cache

from langsmith import traceable
from sentence_transformers import CrossEncoder
from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.app.config import settings
from backend.app.models.models import Chunk, Document
from backend.app.services import embeddings


@dataclass
class RetrievedChunk:
    chunk_id: int
    document_id: int
    short_name: str
    law_name: str
    bab: str | None
    pasal: str
    is_penjelasan: bool
    text: str       # raw Pasal text — used for citation display/verification
    passage: str    # breadcrumb + text — ready to feed the LLM as context
    score: float    # cosine similarity, or reranker score when reranking is on


@lru_cache(maxsize=1)
def _reranker() -> CrossEncoder:
    return CrossEncoder(settings.reranker_model)


@traceable
def search(
    db: Session,
    query: str,
    top_k: int | None = None,
    short_name: str | None = None,
    law_type: str | None = None,
) -> list[RetrievedChunk]:
    top_k = top_k or settings.retrieval_top_k
    query_vector = embeddings.embed_query(query)

    fetch_n = settings.rerank_candidates if settings.reranker_enabled else top_k
    distance = Chunk.embedding.cosine_distance(query_vector)

    stmt = (
        select(Chunk, Document, distance.label("distance"))
        .join(Document, Chunk.document_id == Document.id)
        .where(Chunk.embedding.is_not(None))
    )
    if short_name:
        stmt = stmt.where(Document.short_name == short_name)
    if law_type:
        stmt = stmt.where(Document.law_type == law_type)
    stmt = stmt.order_by(distance).limit(fetch_n)

    results: list[RetrievedChunk] = []
    for chunk, doc, dist in db.execute(stmt).all():
        passage = embeddings.build_passage_text(
            doc.short_name,
            chunk.bab,
            chunk.bab_title,
            chunk.pasal,
            chunk.is_penjelasan,
            chunk.text,
        )
        results.append(
            RetrievedChunk(
                chunk_id=chunk.id,
                document_id=doc.id,
                short_name=doc.short_name,
                law_name=doc.law_name,
                bab=chunk.bab,
                pasal=chunk.pasal,
                is_penjelasan=chunk.is_penjelasan,
                text=chunk.text,
                passage=passage,
                score=1.0 - float(dist),
            )
        )

    if settings.reranker_enabled and results:
        scores = _reranker().predict([(query, r.passage) for r in results])
        for result, score in zip(results, scores):
            result.score = float(score)
        results.sort(key=lambda r: r.score, reverse=True)

    return results[:top_k]
