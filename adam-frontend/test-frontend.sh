#!/bin/bash

echo "Testing ADAM v2 React Frontend..."

# Check if backend is running
if ! curl -s http://localhost:8000/api/health > /dev/null; then
    echo "❌ Backend is not running. Please start it first."
    exit 1
fi

echo "✅ Backend is running"

# Check if frontend is running
if ! curl -s http://localhost:5173 > /dev/null; then
    echo "❌ Frontend is not running. Please start it first."
    exit 1
fi

echo "✅ Frontend is running"

# Test API endpoints
echo -e "\nTesting API endpoints..."

# Test projects endpoint
PROJECTS=$(curl -s http://localhost:8000/api/projects)
echo "Projects response: ${PROJECTS:0:100}..."

echo -e "\n✨ All basic checks passed!"
echo "You can now open http://localhost:5173 in your browser"