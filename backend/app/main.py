from contextlib import asynccontextmanager

from fastapi import FastAPI

from backend.app.config import settings
from backend.app.database import init_db
from backend.app.routers import chat, documents, health


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    yield


app = FastAPI(title=settings.app_name, version=settings.app_version, lifespan=lifespan)

app.include_router(health.router)
app.include_router(documents.router)
app.include_router(chat.router)
