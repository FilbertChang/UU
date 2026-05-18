from datetime import datetime, timezone

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    JSON,
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
)
from sqlalchemy.orm import relationship

from backend.app.config import settings
from backend.app.database import Base


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Document(Base):
    __tablename__ = "documents"

    id = Column(Integer, primary_key=True)
    filename = Column(String(512), nullable=False)
    law_type = Column(String(64), nullable=False, index=True)
    law_number = Column(String(64))
    law_year = Column(Integer)
    law_name = Column(String(512), nullable=False)
    short_name = Column(String(128), nullable=False, index=True)
    # SHA-256 of the normalized extracted text — blocks re-ingesting the same document.
    source_hash = Column(String(64), nullable=False, unique=True)
    uploaded_at = Column(DateTime(timezone=True), nullable=False, default=_utcnow)

    chunks = relationship(
        "Chunk",
        back_populates="document",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )


class Chunk(Base):
    __tablename__ = "chunks"

    id = Column(Integer, primary_key=True)
    document_id = Column(
        Integer,
        ForeignKey("documents.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    chunk_index = Column(Integer, nullable=False)

    bab = Column(String(64))
    bab_title = Column(String(512))
    pasal = Column(String(32), nullable=False, index=True)
    ayat = Column(String(32))
    is_penjelasan = Column(Boolean, nullable=False, default=False)

    text = Column(Text, nullable=False)
    # Populated in Stage 3 (embedding). Nullable until then.
    embedding = Column(Vector(settings.embedding_dim))

    document = relationship("Document", back_populates="chunks")


class ChatSession(Base):
    __tablename__ = "chat_sessions"

    id = Column(String(36), primary_key=True)  # UUID
    created_at = Column(DateTime(timezone=True), nullable=False, default=_utcnow)

    messages = relationship(
        "ChatMessage",
        back_populates="session",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )


class ChatMessage(Base):
    __tablename__ = "chat_messages"

    id = Column(Integer, primary_key=True)
    session_id = Column(
        String(36),
        ForeignKey("chat_sessions.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    role = Column(String(16), nullable=False)  # "user" | "assistant"
    content = Column(Text, nullable=False)
    # Assistant messages only: list of citation dicts and the confidence score.
    citations = Column(JSON)
    confidence = Column(Float)
    created_at = Column(DateTime(timezone=True), nullable=False, default=_utcnow)

    session = relationship("ChatSession", back_populates="messages")
