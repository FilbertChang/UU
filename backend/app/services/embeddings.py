"""Embedding model wrapper for the intfloat/multilingual-e5-* family.

e5 models require an instruction prefix: "query: " for search queries and
"passage: " for indexed documents. Embeddings are L2-normalized so cosine
similarity (pgvector's `<=>`) behaves as expected.

The model is loaded lazily on first use and cached for the process lifetime.
"""

from functools import lru_cache

from sentence_transformers import SentenceTransformer

from backend.app.config import settings


@lru_cache(maxsize=1)
def _model() -> SentenceTransformer:
    return SentenceTransformer(settings.embedding_model)


def build_passage_text(
    short_name: str,
    bab: str | None,
    bab_title: str | None,
    pasal: str,
    is_penjelasan: bool,
    text: str,
) -> str:
    """Prepend a hierarchy breadcrumb so the embedding captures context."""
    crumbs = [short_name]
    if bab:
        crumbs.append(f"Bab {bab} {bab_title}".strip() if bab_title else f"Bab {bab}")
    crumbs.append(f"Penjelasan Pasal {pasal}" if is_penjelasan else f"Pasal {pasal}")
    return " > ".join(crumbs) + "\n\n" + text


def embed_passages(passages: list[str]) -> list[list[float]]:
    vectors = _model().encode(
        [f"passage: {p}" for p in passages],
        normalize_embeddings=True,
    )
    return [vector.tolist() for vector in vectors]


def embed_query(query: str) -> list[float]:
    vector = _model().encode(f"query: {query}", normalize_embeddings=True)
    return vector.tolist()
