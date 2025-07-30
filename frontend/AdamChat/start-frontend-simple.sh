#!/bin/bash
# Start ADAM Frontend (Vite only)

echo "🚀 Starting ADAM Frontend (Vite dev server)..."

# Check if node_modules exists
if [ ! -d "node_modules" ]; then
    echo "📦 Installing dependencies..."
    npm install
fi

# Start Vite directly
echo "🌐 Starting frontend on http://localhost:5173"
echo "✅ ADAM backend is already running on port 8000"
echo ""

# Run npm dev (which now runs vite directly)
npm run dev