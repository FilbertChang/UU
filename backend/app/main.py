from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from backend.app.config import settings
from backend.app.database import init_db
from backend.app.routers import chat, documents, health, stats


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    yield


app = FastAPI(title=settings.app_name, version=settings.app_version, lifespan=lifespan)

app.include_router(health.router)
app.include_router(documents.router)
app.include_router(chat.router)
app.include_router(stats.router)

# Static frontend — mounted last so API routes take precedence.
app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")
