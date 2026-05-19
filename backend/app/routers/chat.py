import uuid
from dataclasses import asdict

from fastapi import APIRouter, Depends, HTTPException
from langsmith import traceable
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.app.config import settings
from backend.app.database import get_db
from backend.app.models.models import ChatMessage, ChatSession
from backend.app.services import citation_verifier, generation, retrieval
from backend.app.services.llm import LLMError, get_provider

router = APIRouter(prefix="/chat", tags=["chat"])


class AskRequest(BaseModel):
    question: str
    session_id: str | None = None
    short_name: str | None = None  # optional metadata filter, e.g. "KUHP"
    law_type: str | None = None


def _load_history(db: Session, session_id: str) -> list[tuple[str, str]]:
    limit = settings.conversation_memory_turns * 2
    messages = db.scalars(
        select(ChatMessage)
        .where(ChatMessage.session_id == session_id)
        .order_by(ChatMessage.id.desc())
        .limit(limit)
    ).all()
    return [(m.role, m.content) for m in reversed(messages)]


@traceable(name="chat_pipeline")
def _run_pipeline(
    db: Session,
    question: str,
    history: list[tuple[str, str]],
    short_name: str | None,
    law_type: str | None,
):
    """Retrieval -> generation -> verification, traced as one nested unit."""
    provider = get_provider()
    search_query = generation.rewrite_query(provider, history, question)
    chunks = retrieval.search(db, search_query, short_name=short_name, law_type=law_type)
    answer = generation.generate_answer(provider, question, chunks, history)
    verification = citation_verifier.verify(db, answer, chunks)
    return chunks, answer, verification


@router.post("/ask")
def ask(payload: AskRequest, db: Session = Depends(get_db)):
    question = payload.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="question must not be empty")

    if payload.session_id:
        session = db.get(ChatSession, payload.session_id)
        if session is None:
            raise HTTPException(status_code=404, detail="session not found")
    else:
        session = ChatSession(id=str(uuid.uuid4()))
        db.add(session)
        db.flush()

    history = _load_history(db, session.id)

    try:
        chunks, answer, verification = _run_pipeline(
            db, question, history, payload.short_name, payload.law_type
        )
    except LLMError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    citations = [asdict(c) for c in verification.citations]

    retrieval_score = sum(c.score for c in chunks) / len(chunks) if chunks else 0.0
    confidence = citation_verifier.compute_confidence(
        retrieval_score, verification.verified_ratio, verification.grounding_ratio
    )

    db.add(ChatMessage(session_id=session.id, role="user", content=question))
    db.add(
        ChatMessage(
            session_id=session.id,
            role="assistant",
            content=answer,
            citations=citations,
            confidence=confidence,
        )
    )
    db.commit()

    return {
        "session_id": session.id,
        "answer": answer,
        "citations": citations,
        "confidence": confidence,
        "verification": {
            "verified_ratio": verification.verified_ratio,
            "grounding_ratio": verification.grounding_ratio,
            "unsupported_claims": verification.unsupported_claims,
            "notes": verification.notes,
        },
        "disclaimer": generation.DISCLAIMER,
    }


@router.get("/history")
def history(session_id: str, db: Session = Depends(get_db)):
    session = db.get(ChatSession, session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="session not found")

    messages = db.scalars(
        select(ChatMessage)
        .where(ChatMessage.session_id == session_id)
        .order_by(ChatMessage.id)
    ).all()
    return {
        "session_id": session_id,
        "messages": [
            {
                "role": m.role,
                "content": m.content,
                "citations": m.citations,
                "confidence": m.confidence,
                "created_at": m.created_at.isoformat(),
            }
            for m in messages
        ],
    }
