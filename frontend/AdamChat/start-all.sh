#!/bin/bash

# Start ADAM web services
echo "🚀 Starting ADAM Web Interface..."
echo ""

# Function to kill processes on exit
cleanup() {
    echo ""
    echo "🛑 Shutting down services..."
    
    # Kill Python backend if running
    if [ ! -z "$PYTHON_PID" ]; then
        kill $PYTHON_PID 2>/dev/null
    fi
    
    # Kill Node server if running
    if [ ! -z "$NODE_PID" ]; then
        kill $NODE_PID 2>/dev/null
    fi
    
    exit 0
}

# Set up trap to cleanup on exit
trap cleanup EXIT INT TERM

# Start Python backend (ADAM v2 API)
echo "📦 Starting Python backend on port 8000..."
cd ../../src/adam_v2
python main.py > ../../frontend/AdamChat/backend.log 2>&1 &
PYTHON_PID=$!
echo "   PID: $PYTHON_PID"

# Wait a bit for Python to start
sleep 3

# Check if Python backend started successfully
if ! kill -0 $PYTHON_PID 2>/dev/null; then
    echo "❌ Python backend failed to start. Check backend.log for errors."
    echo "   Make sure you have installed Python dependencies:"
    echo "   cd src/adam_v2 && pip install -r requirements.txt"
    exit 1
fi

echo "✅ Python backend started successfully"
echo ""

# Start Node.js frontend server
echo "🌐 Starting frontend server on port 5173..."
cd ../../frontend/AdamChat

# Check if node_modules exists
if [ ! -d "node_modules" ]; then
    echo "📦 Installing Node dependencies..."
    npm install
fi

# Start the dev server
npm run dev &
NODE_PID=$!

echo ""
echo "✅ All services started!"
echo ""
echo "📌 Access the web interface at: http://localhost:5173"
echo ""
echo "🔧 Tips for using conversation management:"
echo "   • Hover over conversations to see edit/delete buttons (desktop)"
echo "   • Click the ⋯ menu button on mobile devices"
echo "   • Click the pencil icon to rename conversations"
echo "   • Click the trash icon to delete conversations"
echo ""
echo "Press Ctrl+C to stop all services"
echo ""

# Wait for services
wait