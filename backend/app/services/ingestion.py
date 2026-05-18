"""PDF ingestion for Indonesian legal documents.

Parses the regular hierarchy of Indonesian legislation (Bab > Pasal) into
one chunk per Pasal. The Penjelasan (elucidation) section is detected and its
articles are flagged so they are not confused with the operative text.

Known V1 limitations:
- A Pasal is never split, even if very long (it stays one chunk).
- "Buku" and "Bagian" groupings are not tracked as separate metadata.
- Header/footer cleanup only strips bare page-number lines.
"""

import hashlib
import re
from dataclasses import dataclass
from typing import BinaryIO

import pdfplumber

# A Pasal marker is a line that is *only* "Pasal <n>" — cross-references like
# "sebagaimana dimaksud dalam Pasal 5" appear mid-line and are not matched.
_RE_PASAL = re.compile(r"^Pasal\s+(\d+[A-Za-z]?)\s*$", re.IGNORECASE)
_RE_BAB = re.compile(r"^BAB\s+([IVXLCDM]+)\b\.?\s*(.*)$", re.IGNORECASE)
# Tolerant of letter-spaced headings ("P E N J E L A S A N") seen in some PDFs.
_RE_PENJELASAN = re.compile(r"^P\s*E\s*N\s*J\s*E\s*L\s*A\s*S\s*A\s*N\b", re.IGNORECASE)
_RE_PAGE_NUMBER = re.compile(r"^-?\s*\d+\s*-?$")


@dataclass
class ParsedChunk:
    bab: str | None
    bab_title: str | None
    pasal: str
    ayat: str | None
    is_penjelasan: bool
    text: str


def extract_text(file: BinaryIO) -> str:
    pages: list[str] = []
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            pages.append(page.extract_text() or "")
    return "\n".join(pages)


def _normalize(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip().lower()


def compute_source_hash(text: str) -> str:
    return hashlib.sha256(_normalize(text).encode("utf-8")).hexdigest()


def _clean_lines(text: str) -> list[str]:
    return [ln for ln in text.split("\n") if not _RE_PAGE_NUMBER.match(ln.strip())]


def parse_document(text: str) -> list[ParsedChunk]:
    chunks: list[ParsedChunk] = []

    cur_bab: str | None = None
    cur_bab_title: str | None = None
    in_penjelasan = False

    cur_pasal: str | None = None
    buf: list[str] = []
    expect_bab_title = False

    def flush() -> None:
        nonlocal cur_pasal, buf
        if cur_pasal is not None:
            body = "\n".join(buf).strip()
            if body:
                chunks.append(
                    ParsedChunk(
                        bab=cur_bab,
                        bab_title=cur_bab_title,
                        pasal=cur_pasal,
                        ayat=None,
                        is_penjelasan=in_penjelasan,
                        text=body,
                    )
                )
        cur_pasal, buf = None, []

    for raw in _clean_lines(text):
        line = raw.strip()

        if not line:
            if cur_pasal is not None:
                buf.append("")
            continue

        if expect_bab_title:
            expect_bab_title = False
            is_marker = bool(
                _RE_BAB.match(line)
                or _RE_PASAL.match(line)
                or _RE_PENJELASAN.match(line)
            )
            if not is_marker:
                cur_bab_title = line
                continue
            # otherwise fall through and handle the marker below

        m_bab = _RE_BAB.match(line)
        if m_bab:
            flush()
            cur_bab = m_bab.group(1).upper()
            trailing = m_bab.group(2).strip()
            cur_bab_title = trailing or None
            expect_bab_title = not trailing
            continue

        if _RE_PENJELASAN.match(line):
            flush()
            in_penjelasan = True
            cur_bab = None
            cur_bab_title = None
            expect_bab_title = False
            continue

        m_pasal = _RE_PASAL.match(line)
        if m_pasal:
            flush()
            cur_pasal = m_pasal.group(1)
            continue

        if cur_pasal is not None:
            buf.append(line)
        # text before the first Pasal (title block, Menimbang/Mengingat) is dropped

    flush()
    return chunks
