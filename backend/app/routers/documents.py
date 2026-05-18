import io

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from backend.app.database import get_db
from backend.app.models.models import Chunk, Document
from backend.app.services import embeddings, ingestion

router = APIRouter(prefix="/documents", tags=["documents"])


def _summary(db: Session, doc: Document) -> dict:
    chunk_count = db.scalar(
        select(func.count(Chunk.id)).where(Chunk.document_id == doc.id)
    )
    return {
        "id": doc.id,
        "filename": doc.filename,
        "law_type": doc.law_type,
        "law_number": doc.law_number,
        "law_year": doc.law_year,
        "law_name": doc.law_name,
        "short_name": doc.short_name,
        "chunk_count": chunk_count or 0,
        "uploaded_at": doc.uploaded_at.isoformat(),
    }


@router.post("/upload")
def upload_document(
    file: UploadFile = File(...),
    law_type: str = Form(...),
    law_name: str = Form(...),
    short_name: str = Form(...),
    law_number: str | None = Form(None),
    law_year: int | None = Form(None),
    db: Session = Depends(get_db),
):
    if not (file.filename or "").lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="File must be a PDF")

    try:
        text = ingestion.extract_text(io.BytesIO(file.file.read()))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Failed to read PDF: {exc}")

    if not text.strip():
        raise HTTPException(
            status_code=422,
            detail="No extractable text found — the PDF may be scanned and require OCR",
        )

    source_hash = ingestion.compute_source_hash(text)
    existing = db.scalar(select(Document).where(Document.source_hash == source_hash))
    if existing is not None:
        return {"already_existed": True, "document": _summary(db, existing)}

    parsed = ingestion.parse_document(text)
    if not parsed:
        raise HTTPException(
            status_code=422, detail="No 'Pasal' articles detected in the document"
        )

    doc = Document(
        filename=file.filename,
        law_type=law_type,
        law_number=law_number,
        law_year=law_year,
        law_name=law_name,
        short_name=short_name,
        source_hash=source_hash,
    )
    db.add(doc)
    db.flush()

    seen: set[tuple[str, bool]] = set()
    chunk_objs: list[Chunk] = []
    for parsed_chunk in parsed:
        key = (parsed_chunk.pasal, parsed_chunk.is_penjelasan)
        if key in seen:
            continue
        seen.add(key)
        chunk_objs.append(
            Chunk(
                document_id=doc.id,
                chunk_index=len(chunk_objs),
                bab=parsed_chunk.bab,
                bab_title=parsed_chunk.bab_title,
                pasal=parsed_chunk.pasal,
                ayat=parsed_chunk.ayat,
                is_penjelasan=parsed_chunk.is_penjelasan,
                text=parsed_chunk.text,
            )
        )

    passages = [
        embeddings.build_passage_text(
            doc.short_name, c.bab, c.bab_title, c.pasal, c.is_penjelasan, c.text
        )
        for c in chunk_objs
    ]
    for chunk_obj, vector in zip(chunk_objs, embeddings.embed_passages(passages)):
        chunk_obj.embedding = vector

    db.add_all(chunk_objs)
    db.commit()
    db.refresh(doc)
    return {"already_existed": False, "document": _summary(db, doc)}


@router.get("/list")
def list_documents(db: Session = Depends(get_db)):
    docs = db.scalars(select(Document).order_by(Document.uploaded_at.desc())).all()
    return {"documents": [_summary(db, doc) for doc in docs]}


@router.delete("/{document_id}", status_code=204)
def delete_document(document_id: int, db: Session = Depends(get_db)):
    doc = db.get(Document, document_id)
    if doc is None:
        raise HTTPException(status_code=404, detail="Document not found")
    db.delete(doc)
    db.commit()
