from fastapi import APIRouter, Depends
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from backend.app.database import get_db
from backend.app.models.models import ChatMessage, Chunk, Document

router = APIRouter(tags=["stats"])


@router.get("/stats")
def stats(db: Session = Depends(get_db)):
    by_type = db.execute(
        select(Document.law_type, func.count(Document.id))
        .group_by(Document.law_type)
        .order_by(func.count(Document.id).desc())
    ).all()
    return {
        "total_documents": db.scalar(select(func.count(Document.id))) or 0,
        "total_chunks": db.scalar(select(func.count(Chunk.id))) or 0,
        "total_queries": db.scalar(
            select(func.count(ChatMessage.id)).where(ChatMessage.role == "user")
        )
        or 0,
        "documents_by_type": [
            {"law_type": law_type, "count": count} for law_type, count in by_type
        ],
    }
