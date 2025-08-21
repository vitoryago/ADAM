"""
Main FastAPI application for ADAM v2.0 API
"""

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from contextlib import asynccontextmanager
import logging
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from parent ADAM .env if it exists
parent_env = Path(__file__).parent.parent.parent / ".env"
if parent_env.exists():
    load_dotenv(parent_env)
    logging.info(f"Loaded environment from {parent_env}")

# Load local .env to override if needed
load_dotenv()

# Import database
from database import init_db, close_db

# Import routers
from routers import projects, conversations, messages, memories, voice, voice_streaming, tools

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Starting ADAM v2.0 API...")
    # Initialize database
    await init_db()
    logger.info("Database initialized")
    
    yield
    
    # Shutdown
    logger.info("Shutting down ADAM v2.0 API...")
    # Cleanup resources
    await close_db()


# Create FastAPI app
app = FastAPI(
    title="ADAM v2.0 API",
    description="Project-Based AI Assistant with Memory Isolation - REST API",
    version="2.0.0",
    lifespan=lifespan,
    docs_url="/api/docs",
    redoc_url="/api/redoc"
)

# Configure CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173", "http://localhost:5174"],  # React dev servers
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH"],
    allow_headers=["*"],
    expose_headers=["*"],
)

# Include routers
app.include_router(projects.router, prefix="/api/projects", tags=["projects"])
app.include_router(conversations.router, prefix="/api", tags=["conversations"])
app.include_router(messages.router, prefix="/api", tags=["messages"])
app.include_router(memories.router, prefix="/api", tags=["memories"])
app.include_router(voice.router, prefix="/api/voice", tags=["voice"])
app.include_router(voice_streaming.router, tags=["voice-streaming"])
app.include_router(tools.router, prefix="/api/tools", tags=["tools"])


@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "version": "2.0.0", "type": "api"}


@app.get("/api/debug/cors")
async def debug_cors(request: Request):
    """Debug CORS headers"""
    return {
        "origin": request.headers.get("origin"),
        "headers": dict(request.headers),
        "url": str(request.url)
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )