"""
Main FastAPI application for ADAM v2.0
"""

from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from contextlib import asynccontextmanager
import logging
import os
from pathlib import Path

# Import database
from database import init_db, close_db

# Import routers
from routers import projects, conversations

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Starting ADAM v2.0...")
    # Initialize database
    await init_db()
    logger.info("Database initialized")
    
    yield
    
    # Shutdown
    logger.info("Shutting down ADAM v2.0...")
    # Cleanup resources
    await close_db()


# Create FastAPI app
app = FastAPI(
    title="ADAM v2.0",
    description="Project-Based AI Assistant with Memory Isolation",
    version="2.0.0",
    lifespan=lifespan
)

# Create directories if they don't exist
static_dir = Path("static")
static_dir.mkdir(exist_ok=True)
templates_dir = Path("templates")
templates_dir.mkdir(exist_ok=True)

# Mount static files
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

# Templates
templates = Jinja2Templates(directory=str(templates_dir))

# Include routers
app.include_router(projects.router, prefix="/api/projects", tags=["projects"])
app.include_router(conversations.router, prefix="/api", tags=["conversations"])


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Main dashboard page"""
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/project/{project_id}", response_class=HTMLResponse)
async def project_view(request: Request, project_id: str):
    """Project conversation view"""
    return templates.TemplateResponse("conversation.html", {
        "request": request,
        "project_id": project_id
    })


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "version": "2.0.0"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )