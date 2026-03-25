"""
ADAM API - Single FastAPI entrypoint.

Unified API server for ADAM 4.0 (Analytics Data Assistant with Memory).
Consolidated from src/adam_v2/main.py.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import logging
import os

# Fix tokenizers warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan: startup and shutdown hooks."""
    # Startup
    logger.info("Starting ADAM 4.0 API...")
    try:
        from adam.api.models import Base
        from adam.database import get_engine
        engine = get_engine()
        await engine.create_tables()
        logger.info("Database initialized")
    except Exception as e:
        logger.error(f"Error initializing database: {e}")

    yield

    # Shutdown
    logger.info("Shutting down ADAM 4.0 API...")
    try:
        from adam.database import close_db
        await close_db()
    except Exception as e:
        logger.error(f"Error closing database: {e}")


# Create FastAPI app
app = FastAPI(
    title="ADAM - Analytics Data Assistant with Memory",
    version="4.0.0",
    description="AI assistant with persistent memory and intelligent routing",
    lifespan=lifespan,
    docs_url="/api/docs",
    redoc_url="/api/redoc",
)

# CORS - allow all origins for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

# Include routers
from adam.api.routers import (
    conversations, projects, memories, messages,
    voice, voice_streaming, lineage, styles, dbt,
    deep_discussion,
)

app.include_router(projects.router, prefix="/api/projects", tags=["projects"])
app.include_router(conversations.router, prefix="/api", tags=["conversations"])
app.include_router(messages.router, prefix="/api", tags=["messages"])
app.include_router(memories.router, prefix="/api", tags=["memories"])
app.include_router(voice.router, prefix="/api/voice", tags=["voice"])
app.include_router(voice_streaming.router, prefix="/api/voice/stream", tags=["voice-streaming"])
app.include_router(lineage.router, prefix="/api/lineage", tags=["lineage"])
app.include_router(styles.router, prefix="/api", tags=["styles"])
app.include_router(dbt.router, prefix="/api/dbt", tags=["dbt"])
app.include_router(deep_discussion.router, prefix="/api/deep-discussion", tags=["deep-discussion"])


@app.get("/health")
async def health_check():
    """Health check endpoint for monitoring."""
    return {"status": "ok", "version": "4.0.0"}


@app.get("/api/health")
async def api_health_check():
    """API health check endpoint."""
    return {"status": "healthy", "version": "4.0.0", "type": "api"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "adam.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
