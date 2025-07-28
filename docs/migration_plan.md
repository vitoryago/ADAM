# ADAM Migration Plan: v2 Features to v1

## Summary
The user wants to go back to ADAM v1 (Streamlit) because:
1. Markdown rendering works properly
2. Code snippets display correctly
3. It's a proven, working solution

## ADAM Identity Issue Fixed
Fixed in v2 by adding system prompt: "You are ADAM (Advanced Data Analytics Model), an AI assistant specializing in software development, data analysis, and problem-solving."

## Key Features to Migrate from v2 to v1

### 1. Project-Based Organization
- **v2 Feature**: Each project has its own isolated memory space
- **Implementation for v1**: 
  - Add project selector dropdown in sidebar
  - Store project_id in session state
  - Modify memory initialization to use project-specific collections

### 2. Multiple Conversations per Project
- **v2 Feature**: Users can have multiple conversation threads
- **Implementation for v1**:
  - Add conversation management in sidebar
  - Store conversations in session state
  - Allow switching between conversations

### 3. Better Cost Tracking
- **v2 Feature**: Track costs per message, conversation, and project
- **Implementation for v1**:
  - v1 already has basic cost tracking
  - Enhance with per-project cost summaries

### 4. SSE Streaming (Optional)
- **v2 Feature**: Real-time streaming responses
- **Implementation for v1**:
  - Streamlit has built-in streaming support with st.write_stream()

## Files to Keep from v2

### Backend Services (can be integrated into v1):
- `src/adam_v2/services/advanced_memory_service.py` - Better memory evaluation
- `src/adam_v2/services/memory_service.py` - Project-based memory isolation
- `src/adam_v2/models.py` - Database models (if we want to add persistence)

## Files to Remove

### React Frontend (not working properly):
- `/adam-frontend/` - entire folder can be removed
- `src/adam_v2/backup_ui_files/` - HTMX attempts, keep for reference only

### Unused v2 files:
- `src/adam_v2/routers/` - FastAPI routers (not needed for Streamlit)
- `src/adam_v2/main.py` - FastAPI app (not needed)

## Recommended Approach

1. **Keep using v1 Streamlit app** (`web/adam_web.py`)
2. **Add project support** by modifying the sidebar
3. **Integrate advanced memory service** from v2
4. **Keep the fixed LLM service** with ADAM identity

## Quick Start

To run ADAM v1 with Streamlit:
```bash
cd /Users/vitoryago/ADAM
pip install -r requirements_web.txt
cd web
python adam_web.py
```

The Streamlit app will open at http://localhost:8501 with working markdown and code snippets.