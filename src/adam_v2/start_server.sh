#!/bin/bash
# Start ADAM v2 with the correct Python environment

echo "🚀 Starting ADAM v2.0..."

# Use the ADAM virtual environment Python directly
PYTHON_PATH="/Users/vitoryago/ADAM/venv/bin/python"

if [ ! -f "$PYTHON_PATH" ]; then
    echo "❌ Error: Python virtual environment not found at $PYTHON_PATH"
    exit 1
fi

echo "📍 Using Python: $PYTHON_PATH"
echo "📂 Working directory: $(pwd)"

# Check if we're in the right directory
if [ ! -f "main.py" ]; then
    echo "❌ Error: main.py not found. Please run from adam_v2 directory"
    exit 1
fi

echo "🌐 Starting server on http://localhost:8000"
echo "📝 Note: Memory features may be limited without ChromaDB"
echo "Press Ctrl+C to stop"
echo ""

# Run the server
$PYTHON_PATH main.py