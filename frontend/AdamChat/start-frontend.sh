#!/bin/bash
# Start ADAM Frontend

echo "🚀 Starting ADAM Frontend..."

# Check if node_modules exists
if [ ! -d "node_modules" ]; then
    echo "📦 Installing dependencies..."
    npm install
fi

# Start the Vite development server only (not the Node.js backend)
echo "🌐 Starting frontend on http://localhost:5173"
echo "📝 Note: Make sure ADAM backend is running on port 8000"
cd client && npm run dev