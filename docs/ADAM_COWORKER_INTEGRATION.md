# ADAM Coworker Integration Guide

## Overview

This guide explains how to integrate the new coworker features into ADAM, transforming it from a simple chatbot into a true AI coworker that can see your screen and maintain project-based memories.

## New Components

### 1. Project Manager (`src/adam/project_manager.py`)
- Manages multiple projects with isolated memory spaces
- Each project has its own ChromaDB collection
- Tracks project metadata and statistics

### 2. Screen Capture Service (`src/adam/screen_capture.py`)
- Cross-platform screen capture (full screen, window, region)
- OCR text extraction from screenshots
- Automatic change detection for monitoring
- Integration with vision models

### 3. Project-Aware Memory (`src/adam/project_aware_memory.py`)
- Extends base memory system with project isolation
- Automatic screen capture integration
- Project-specific search and retrieval
- Screen monitoring capabilities

## Installation

```bash
# Install additional dependencies
pip install -r requirements_coworker.txt

# For OCR support, also install Tesseract
# macOS: brew install tesseract
# Ubuntu: sudo apt-get install tesseract-ocr
# Windows: Download from https://github.com/UB-Mannheim/tesseract/wiki
```

## Quick Start

### 1. Basic Usage

```python
from adam.project_aware_memory import ProjectAwareMemory
from adam.llm.client import UnifiedLLMClient

# Initialize project-aware memory
memory = ProjectAwareMemory()

# Create a new project
project = memory.create_project("My Python Project", "Working on data analysis")

# Capture and analyze screen
screen_data = memory.screen_capture.capture_screen()
if screen_data:
    # Store with screen context
    memory.remember_with_screen(
        query="What's on my screen?",
        response="I can see you're working on...",
        screen_capture=screen_data
    )
```

### 2. Project Switching

```python
# List all projects
projects = memory.list_projects()

# Switch to a different project
memory.switch_project(project_id)

# All subsequent memories will be stored in this project
```

### 3. Screen Monitoring

```python
# Start monitoring screen for changes
memory.start_screen_monitoring(interval=30)  # Check every 30 seconds

# Stop monitoring
memory.stop_screen_monitoring()
```

## Integration with Existing ADAM

### Option 1: Drop-in Replacement

Replace `ADAMMemoryAdvanced` with `ProjectAwareMemory` in your code:

```python
# Before
from adam.memory import ADAMMemoryAdvanced
memory = ADAMMemoryAdvanced()

# After
from adam.project_aware_memory import ProjectAwareMemory
memory = ProjectAwareMemory()
```

### Option 2: Gradual Integration

Use alongside existing memory system:

```python
# Keep existing memory for backward compatibility
base_memory = ADAMMemoryAdvanced()

# Add project memory for new features
project_memory = ProjectAwareMemory()
```

## API Integration

For your frontend to connect, you'll need these endpoints:

```python
# Example FastAPI endpoints
@app.post("/api/projects/create")
async def create_project(name: str, description: str):
    project = memory.create_project(name, description)
    return {"id": project.id, "name": project.name}

@app.get("/api/projects/list")
async def list_projects():
    return memory.list_projects()

@app.post("/api/projects/{project_id}/activate")
async def activate_project(project_id: str):
    success = memory.switch_project(project_id)
    return {"success": success}

@app.post("/api/capture/screen")
async def capture_screen():
    image_data = memory.screen_capture.capture_screen()
    if image_data:
        # Analyze with vision model
        response = await llm_client.complete(
            prompt="What do you see?",
            model="grok-2-vision-1212",
            image_data=image_data
        )
        # Store in memory
        memory_id = memory.remember_with_screen(
            query="Screen capture",
            response=response.content,
            screen_capture=image_data
        )
        return {"memory_id": memory_id, "analysis": response.content}
    return {"error": "Screen capture failed"}
```

## Use Cases

### 1. Development Assistant
- Capture error messages automatically
- Store debugging sessions per project
- Quick access to project-specific solutions

### 2. Learning Companion
- Track what you're studying per subject
- Capture and annotate lecture slides
- Build knowledge base by topic

### 3. Research Assistant
- Organize findings by research project
- Capture and analyze data visualizations
- Maintain separate contexts for different papers

## Advanced Features

### Custom Screen Analysis

```python
from adam.screen_capture import ScreenContextAnalyzer

analyzer = ScreenContextAnalyzer(memory.screen_capture)
context = analyzer.analyze_screen_context(screen_data)

if "error" in context.get("extracted_text", "").lower():
    # Proactively offer help
    print("I noticed an error on your screen. Would you like help?")
```

### Project Templates

```python
# Create project with predefined settings
project = memory.create_project(
    name="Django Web App",
    description="Building a web application with Django"
)
project.settings = {
    "preferred_model": "grok-4",
    "auto_capture_errors": True,
    "monitor_interval": 60
}
```

## Performance Considerations

1. **Screen Capture**: 
   - Full screen capture uses ~5-10MB per image
   - Consider using region capture for specific areas
   - OCR adds ~1-2 seconds processing time

2. **Memory Isolation**:
   - Each project has separate vector space
   - Switching projects is fast (< 100ms)
   - Search only queries active project by default

3. **Monitoring**:
   - Adjust interval based on needs (5s for active help, 60s for passive)
   - Change detection reduces unnecessary captures
   - Consider CPU usage on older machines

## Troubleshooting

### Screen Capture Not Working
- **macOS**: Grant screen recording permission in System Preferences
- **Linux**: May need to run with appropriate permissions
- **Windows**: Some applications may block capture

### OCR Not Working
- Install Tesseract: `brew install tesseract` (macOS)
- Check language data files are installed
- For better accuracy, pre-process images (contrast, resolution)

### Project Switching Issues
- Ensure ChromaDB has write permissions
- Check disk space for vector storage
- Verify collection names are valid

## Future Enhancements

1. **Auto-Project Detection**: Detect project based on active window/directory
2. **Voice Integration**: "Hey ADAM, what's on my screen?"
3. **Collaborative Features**: Share project memories with team
4. **Plugin System**: Integrate with IDEs, browsers, and other tools

## Example: Complete Coworker Setup

See `examples/coworker_demo.py` for a full working example that demonstrates:
- Project creation and management
- Screen capture and analysis
- Project-specific memory storage
- Cross-project search
- Continuous monitoring

This integration makes ADAM a true AI coworker that understands your context and provides project-aware assistance!