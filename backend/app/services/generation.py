"""Prompt construction and LLM answer generation.

The system prompt enforces the anti-hallucination rules: answer only from the
supplied context, always cite a Pasal, admit when the answer is not in the
context, and refuse case-specific advice.
"""

from backend.app.services.llm import LLMProvider
from backend.app.services.retrieval import RetrievedChunk

DISCLAIMER = (
    "Informasi ini bersifat edukatif dan tidak menggantikan konsultasi dengan "
    "pengacara profesional."
)

_SYSTEM_PROMPT = """Anda adalah asisten hukum untuk peraturan perundang-undangan Indonesia.
Jawab pertanyaan HANYA berdasarkan KONTEKS pasal yang diberikan.

Aturan wajib:
1. Selalu sertakan rujukan pasal pada jawaban, contoh: "(Pasal 3 UU 8/1999)".
2. Jika KONTEKS tidak memuat informasi yang relevan, katakan dengan jujur bahwa Anda
   tidak menemukan dasar hukumnya dalam dokumen yang tersedia. JANGAN mengarang pasal
   maupun isi hukum.
3. Jangan memberi nasihat untuk kasus pribadi yang spesifik dan jangan memprediksi
   hasil suatu perkara; arahkan pengguna untuk berkonsultasi dengan pengacara.
4. Jawab dalam Bahasa Indonesia yang jelas dan ringkas."""

_REWRITE_SYSTEM = """Anda menulis ulang pertanyaan lanjutan menjadi satu pertanyaan yang
berdiri sendiri dan dapat dipahami tanpa melihat riwayat percakapan. Keluarkan HANYA
pertanyaan hasil tulis ulang, tanpa penjelasan apa pun."""


def _citation_label(chunk: RetrievedChunk) -> str:
    label = f"Pasal {chunk.pasal} {chunk.short_name}"
    return f"Penjelasan {label}" if chunk.is_penjelasan else label


def _format_context(chunks: list[RetrievedChunk]) -> str:
    blocks = [
        f"[{i}] {_citation_label(c)}\n{c.text}" for i, c in enumerate(chunks, start=1)
    ]
    return "\n\n".join(blocks)


def _format_history(history: list[tuple[str, str]]) -> str:
    speakers = {"user": "Pengguna", "assistant": "Asisten"}
    return "\n".join(f"{speakers.get(role, role)}: {content}" for role, content in history)


def rewrite_query(
    provider: LLMProvider, history: list[tuple[str, str]], question: str
) -> str:
    """Condense a follow-up question into a standalone search query."""
    if not history:
        return question
    user = (
        f"Riwayat percakapan:\n{_format_history(history)}\n\n"
        f"Pertanyaan lanjutan: {question}"
    )
    rewritten = provider.complete(_REWRITE_SYSTEM, user).strip()
    # small models sometimes add commentary — keep the first non-empty line
    first_line = next((ln for ln in rewritten.splitlines() if ln.strip()), "")
    return first_line.strip() or question


def generate_answer(
    provider: LLMProvider,
    question: str,
    chunks: list[RetrievedChunk],
    history: list[tuple[str, str]],
) -> str:
    context = _format_context(chunks) if chunks else "(tidak ada konteks yang ditemukan)"
    parts = [f"KONTEKS:\n{context}"]
    if history:
        parts.append(f"RIWAYAT PERCAKAPAN:\n{_format_history(history)}")
    parts.append(f"PERTANYAAN: {question}")
    return provider.complete(_SYSTEM_PROMPT, "\n\n".join(parts)).strip()
