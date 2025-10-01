#!/bin/bash
# Start ADAM Web Interface (Backend + Frontend)

echo "🚀 Starting ADAM Web Interface..."
echo ""

# Check if we're in the ADAM directory
if [ ! -d "src/adam_v2" ]; then
    echo "❌ Error: Must run from ADAM root directory"
    exit 1
fi

# Kill any existing processes on the ports
echo "🔄 Cleaning up existing processes..."
lsof -ti:8000 | xargs kill -9 2>/dev/null
lsof -ti:5173 | xargs kill -9 2>/dev/null

# Start backend
echo ""
echo "📦 Starting Backend (FastAPI) on http://localhost:8000..."
cd src/adam_v2
python main.py > ../../backend.log 2>&1 &
BACKEND_PID=$!
cd ../..

# Wait for backend to start
sleep 3

# Check if backend started
if curl -s http://localhost:8000/api/health > /dev/null; then
    echo "✅ Backend started successfully!"
else
    echo "❌ Backend failed to start. Check backend.log"
    kill $BACKEND_PID 2>/dev/null
    exit 1
fi

# Start frontend
echo ""
echo "🎨 Starting Frontend (React + Vite) on http://localhost:5173..."
cd frontend/AdamChat
npm run dev > ../../frontend.log 2>&1 &
FRONTEND_PID=$!
cd ../..

# Wait for frontend to start
sleep 5

echo ""
echo "✅ ADAM Web Interface started!"
echo ""
echo "📍 URLs:"
echo "   Frontend: http://localhost:5173"
echo "   Backend API: http://localhost:8000"
echo "   API Docs: http://localhost:8000/api/docs"
echo ""
echo "📊 Process IDs:"
echo "   Backend PID: $BACKEND_PID"
echo "   Frontend PID: $FRONTEND_PID"
echo ""
echo "📝 Logs:"
echo "   Backend: tail -f backend.log"
echo "   Frontend: tail -f frontend.log"
echo ""
echo "🛑 To stop: kill $BACKEND_PID $FRONTEND_PID"
echo ""
echo "🎉 Open http://localhost:5173 in your browser!"
