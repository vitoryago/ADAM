# 🧹 ADAM Project Cleanup Plan

## Current State: MESSY! 
- **Size**: 2.3GB (1.4GB is venv alone!)
- **Files**: 23,807 Python files (mostly dependencies)
- **Clutter**: 1,291 `__pycache__` folders
- **Duplication**: Code duplicated in multiple places

## 🗑️ IMMEDIATE DELETIONS (Safe to Delete Now)

### 1. Clean Python Cache (Saves ~200MB)
```bash
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null
find . -name "*.pyc" -delete
find . -name "*.pyo" -delete
```

### 2. Remove venv from repo (Saves 1.4GB!)
```bash
rm -rf venv/
echo "venv/" >> .gitignore
```

### 3. Delete Test Files
```bash
rm test_*.py  # All test files in root
rm test_tools_simple.py
rm test_snowflake_*.py
rm test_new_tools.py
```

### 4. Remove Duplicate ADAM Code
```bash
# This is a COMPLETE DUPLICATE of src/adam!
rm -rf frontend/AdamChat/python_backend/src/adam
```

### 5. Clean Frontend Build Files
```bash
rm -rf frontend/AdamChat/dist/
rm -rf frontend/AdamChat/node_modules/  # If committed by mistake
```

## 📁 FOLDER REORGANIZATION

### Current (Messy):
```
/
├── adam_complete_conversations/   # What is this?
├── adam_memory_advanced/          # Duplicate of src/adam?
├── adam_memory_tools/              # More duplicates?
├── archive/                        # Should be deleted
├── cli/                           # OK
├── config/                        # OK
├── data/                          # Should be in .gitignore
├── docs/                          # OK
├── examples/                      # OK
├── frontend/                      # OK but needs cleanup
├── knowledge/                     # What is this?
├── notebooks/                     # Move to examples/
├── scripts/                       # OK
├── src/                          # OK - main code
├── test_watch_dir/               # Delete
├── tests/                        # OK
├── venv/                         # DELETE!
├── vscode-extension/             # OK
├── web/                          # OK
└── [20+ test files in root]     # DELETE ALL!
```

### Proposed (Clean):
```
/
├── src/                    # All source code
│   ├── adam/              # Core ADAM system
│   ├── adam_v2/           # FastAPI backend
│   └── tools/             # New tools we built
├── frontend/              # All frontends
│   ├── react-app/         # Rename from AdamChat
│   ├── streamlit/         # Move from web/
│   └── cli/               # Move from root
├── extensions/            # IDE integrations
│   └── vscode/           # Move from vscode-extension/
├── tests/                 # All tests
├── docs/                  # Documentation
├── examples/              # Examples & notebooks
├── scripts/               # Utility scripts
├── docker/                # Docker configs
└── .github/               # CI/CD
```

## 🔨 MERGE DUPLICATE FILES

### Memory System (38 files → 5 files)
Consolidate all memory files into:
- `src/adam/memory/core.py` - Main memory system
- `src/adam/memory/network.py` - Memory connections
- `src/adam/memory/lifecycle.py` - Cleanup/compression
- `src/adam/memory/project.py` - Project isolation
- `src/adam/memory/search.py` - Retrieval & RAG

### LLM Integration (Multiple files → 2 files)
- `src/adam/llm/client.py` - Unified client (keep)
- `src/adam/llm/routing.py` - Query analysis & routing

### Tools (Scattered → Organized)
```
src/adam/tools/
├── __init__.py
├── sql/
│   ├── generator.py       # AI SQL generation
│   ├── executor.py        # Snowflake execution
│   └── templates.py       # SQL templates
├── web/
│   └── search.py          # Model-native search
└── code/
    ├── executor.py        # Code execution
    └── generator.py       # File generation
```

## 🧹 CLEANUP COMMANDS

### Phase 1: Delete Obvious Junk (Safe)
```bash
# Clean Python artifacts
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null
find . -name "*.pyc" -delete
find . -name "*.pyo" -delete
find . -name ".DS_Store" -delete

# Remove venv
rm -rf venv/

# Clean test files from root
rm test_*.py

# Remove build artifacts
rm -rf frontend/AdamChat/dist/
rm -rf frontend/AdamChat/build/
rm -rf *.egg-info/

# Clean data files
rm -rf data/adam_memory/  # If you want fresh start
```

### Phase 2: Remove Duplicates
```bash
# Remove duplicate ADAM in frontend
rm -rf frontend/AdamChat/python_backend/src/adam/

# Archive old folders
mkdir -p archive/old_structure
mv adam_complete_conversations/ archive/old_structure/
mv adam_memory_advanced/ archive/old_structure/
mv adam_memory_tools/ archive/old_structure/
mv test_watch_dir/ archive/old_structure/
```

### Phase 3: Reorganize
```bash
# Create new structure
mkdir -p src/adam/memory
mkdir -p src/adam/tools/{sql,web,code}
mkdir -p frontend/react-app
mkdir -p extensions/vscode

# Move files
mv frontend/AdamChat/* frontend/react-app/
mv vscode-extension/* extensions/vscode/
mv web/* frontend/streamlit/
mv cli frontend/

# Consolidate memory files
# (This needs manual merging due to logic)
```

## 📋 Priority Actions

### 1. **URGENT - Add .gitignore entries**
```gitignore
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
venv/
env/
ENV/

# IDE
.vscode/
.idea/
*.swp
*.swo
.DS_Store

# Data
data/
*.db
*.sqlite
adam_memory/

# Logs
*.log
logs/

# Frontend
node_modules/
dist/
build/
*.map

# Environment
.env
.env.local
```

### 2. **Delete venv immediately** (1.4GB!)
```bash
rm -rf venv/
pip freeze > requirements.txt  # Save dependencies first
```

### 3. **Clean all caches**
```bash
# Run this cleanup script
#!/bin/bash
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null
find . -name "*.pyc" -delete
find . -name "*.pyo" -delete
find . -name ".DS_Store" -delete
echo "Cleaned Python caches and OS files"
```

## 📊 Expected Results

### Before Cleanup:
- Size: 2.3GB
- Python files: 23,807
- Folders: Chaotic

### After Cleanup:
- Size: ~200MB
- Python files: ~100 (actual code)
- Folders: Organized

### Space Saved: ~2.1GB (91% reduction!)

## ⚠️ Before You Start

1. **Commit current work**: `git add . && git commit -m "Before cleanup"`
2. **Create backup branch**: `git checkout -b pre-cleanup-backup`
3. **Test after each phase**: Make sure ADAM still works

## 🎯 Final Structure Vision

```
ADAM/
├── src/adam/           # Core brain (memory, LLM, conversation)
├── src/adam_v2/        # API layer (FastAPI)
├── frontend/           # All UIs
├── extensions/         # IDE plugins
├── tests/              # All tests
├── docs/               # Documentation
└── docker/             # Deployment

Clean. Organized. Professional.
```

---

**Ready to clean? Start with Phase 1 - it's completely safe and will free up 1.6GB immediately!**