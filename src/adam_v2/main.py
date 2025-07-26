#!/usr/bin/env python3
"""
ADAM v2.0 - Project-Based Memory System
Main FastAPI application
"""

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uvicorn
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from adam_v2.database import init_db
from adam_v2.routers import projects, conversations, messages
from adam_v2.models import Project, Conversation

app = FastAPI(title="ADAM v2.0", version="2.0.0")

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Templates
templates = Jinja2Templates(directory="templates")

# Include routers
app.include_router(projects.router, prefix="/api/projects", tags=["projects"])
app.include_router(conversations.router, prefix="/api/conversations", tags=["conversations"])
app.include_router(messages.router, prefix="/api/messages", tags=["messages"])

@app.on_event("startup")
async def startup_event():
    """Initialize database on startup"""
    init_db()

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Main page with HTMX interface"""
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/project/{project_id}", response_class=HTMLResponse)
async def project_view(request: Request, project_id: str):
    """Project view page"""
    return templates.TemplateResponse("project.html", {
        "request": request,
        "project_id": project_id
    })

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)