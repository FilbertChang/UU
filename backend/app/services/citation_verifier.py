"""Citation verification — the core anti-hallucination check.

Two layers of checking on the LLM's answer:

1. Citation check — for every "Pasal N" cited, classify it:
   - grounded       — the Pasal was in the retrieved context (legitimately usable)
   - out_of_context — exists in the database but was NOT retrieved (recalled from
                      training data, not from the provided context)
   - hallucinated   — does not exist in the database at all
   Grounded citations also get a token-overlap "text support" score.

2. Claim grounding — every substantive sentence of the answer is checked for token
   overlap against the retrieved context. Sentences with little overlap are flagged
   as unsupported ("padding" — invented elaboration the model added on its own).

Confidence = (0.5 * retrieval similarity + 0.5 * verified-citation ratio)
             with a bounded penalty for low claim-grounding (at most -50%).
"""

import re
from dataclasses import dataclass

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from backend.app.config import settings
from backend.app.models.models import Chunk
from backend.app.services.retrieval import RetrievedChunk

_PASAL_MENTION = re.compile(r"Pasal\s+(\d+[A-Za-z]?)", re.IGNORECASE)
_WORD = re.compile(r"[a-zA-Z]+")
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")
_MIN_CLAIM_TOKENS = 4

# High-frequency Indonesian function words — removed so token overlap reflects content.
_STOPWORDS = frozenset({
    "yang", "dan", "dari", "untuk", "dengan", "pada", "dalam", "adalah", "atau",
    "itu", "ini", "sebagai", "oleh", "akan", "tidak", "juga", "dapat", "ada",
    "karena", "agar", "atas", "para", "suatu", "tersebut", "yaitu", "antara",
    "telah", "bahwa", "sesuai", "serta", "maupun", "jika", "maka", "hal", "secara",
    "setiap", "dimaksud", "ialah", "kepada", "yakni", "dll",
})


@dataclass
class VerifiedCitation:
    pasal: str
    status: str            # "grounded" | "out_of_context" | "hallucinated"
    verified: bool         # grounded AND text_support >= threshold
    text_support: float    # 0..1 — fraction of the Pasal's content words in the answer
    label: str | None      # e.g. "Pasal 4 UU 8/1999" — known for grounded citations
    short_name: str | None
    law_name: str | None
    is_penjelasan: bool
    text: str | None       # original Pasal text — known for grounded citations


@dataclass
class VerificationResult:
    citations: list[VerifiedCitation]
    verified_ratio: float
    grounding_ratio: float
    has_citations: bool
    unsupported_claims: list[str]
    notes: list[str]


def _content_tokens(text: str) -> set[str]:
    return {
        token
        for token in _WORD.findall(text.lower())
        if len(token) > 2 and token not in _STOPWORDS
    }


def _text_support(answer_tokens: set[str], pasal_text: str) -> float:
    pasal_tokens = _content_tokens(pasal_text)
    if not pasal_tokens:
        return 0.0
    return len(pasal_tokens & answer_tokens) / len(pasal_tokens)


def _claim_grounding(
    answer: str, chunks: list[RetrievedChunk]
) -> tuple[float, list[str]]:
    """Fraction of substantive answer sentences supported by the retrieved context."""
    chunk_token_sets = [
        tokens for tokens in (_content_tokens(c.text) for c in chunks) if tokens
    ]
    evaluated = 0
    supported = 0
    unsupported: list[str] = []
    for segment in _SENTENCE_SPLIT.split(answer):
        segment = segment.strip()
        if segment.endswith(":"):
            continue  # a list intro / framing line, not a standalone claim
        seg_tokens = _content_tokens(segment)
        if len(seg_tokens) < _MIN_CLAIM_TOKENS:
            continue  # too short to be a substantive claim
        evaluated += 1
        best = max(
            (len(seg_tokens & cset) / len(seg_tokens) for cset in chunk_token_sets),
            default=0.0,
        )
        if best >= settings.claim_grounding_threshold:
            supported += 1
        else:
            unsupported.append(segment)
    ratio = round(supported / evaluated, 3) if evaluated else 1.0
    return ratio, unsupported


def verify(
    db: Session, answer: str, chunks: list[RetrievedChunk]
) -> VerificationResult:
    mentions: list[str] = []
    seen: set[str] = set()
    for match in _PASAL_MENTION.finditer(answer):
        pasal = match.group(1)
        if pasal.lower() not in seen:
            seen.add(pasal.lower())
            mentions.append(pasal)

    answer_tokens = _content_tokens(answer)
    by_pasal: dict[str, list[RetrievedChunk]] = {}
    for chunk in chunks:
        by_pasal.setdefault(chunk.pasal.lower(), []).append(chunk)

    citations: list[VerifiedCitation] = []
    notes: list[str] = []

    for pasal in mentions:
        context_chunks = by_pasal.get(pasal.lower(), [])
        if context_chunks:
            best = max(
                context_chunks,
                key=lambda c: _text_support(answer_tokens, c.text),
            )
            support = round(_text_support(answer_tokens, best.text), 3)
            verified = support >= settings.citation_support_threshold
            label = f"Pasal {best.pasal} {best.short_name}"
            if best.is_penjelasan:
                label = "Penjelasan " + label
            citations.append(
                VerifiedCitation(
                    pasal=pasal,
                    status="grounded",
                    verified=verified,
                    text_support=support,
                    label=label,
                    short_name=best.short_name,
                    law_name=best.law_name,
                    is_penjelasan=best.is_penjelasan,
                    text=best.text,
                )
            )
            if not verified:
                notes.append(
                    f"Rujukan Pasal {pasal}: isi jawaban kurang cocok dengan teks "
                    f"pasal aslinya (dukungan teks {support:.0%})."
                )
        else:
            exists = (
                db.scalar(
                    select(Chunk.id)
                    .where(func.lower(Chunk.pasal) == pasal.lower())
                    .limit(1)
                )
                is not None
            )
            status = "out_of_context" if exists else "hallucinated"
            citations.append(
                VerifiedCitation(
                    pasal=pasal,
                    status=status,
                    verified=False,
                    text_support=0.0,
                    label=None,
                    short_name=None,
                    law_name=None,
                    is_penjelasan=False,
                    text=None,
                )
            )
            if exists:
                notes.append(
                    f"Pasal {pasal} dirujuk tetapi tidak termasuk konteks yang "
                    f"diambil — tidak dapat diverifikasi."
                )
            else:
                notes.append(
                    f"PERINGATAN: Pasal {pasal} dirujuk tetapi TIDAK ditemukan di "
                    f"basis data — kemungkinan halusinasi."
                )

    verified_count = sum(1 for c in citations if c.verified)
    verified_ratio = round(verified_count / len(mentions), 3) if mentions else 0.0
    if not mentions:
        notes.append("Jawaban tidak memuat rujukan pasal apa pun.")

    grounding_ratio, unsupported_claims = _claim_grounding(answer, chunks)
    if unsupported_claims:
        notes.append(
            f"{len(unsupported_claims)} kalimat jawaban kurang didukung teks pasal — "
            f"kemungkinan tambahan di luar konteks."
        )

    return VerificationResult(
        citations=citations,
        verified_ratio=verified_ratio,
        grounding_ratio=grounding_ratio,
        has_citations=bool(mentions),
        unsupported_claims=unsupported_claims,
        notes=notes,
    )


def compute_confidence(
    retrieval_score: float, verified_ratio: float, grounding_ratio: float
) -> float:
    """Hybrid confidence: half retrieval quality, half verified-citation ratio.

    Claim grounding is a noisy lexical signal, so it only applies a bounded
    penalty — a fully ungrounded answer loses at most half its score, so a stray
    framing sentence cannot crater an otherwise sound answer.
    """
    base = 0.5 * retrieval_score + 0.5 * verified_ratio
    return round(base * (0.5 + 0.5 * grounding_ratio), 3)
