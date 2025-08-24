#!/bin/bash

echo "🚀 Starting ADAM Services..."

# Check if Redis is installed
if ! command -v redis-server &> /dev/null; then
    echo "⚠️  Redis not installed. Installing with Homebrew..."
    brew install redis
fi

# Start Redis in background
echo "📦 Starting Redis..."
redis-server --daemonize yes

# Wait for Redis to start
sleep 2

# Check if Redis is running
if redis-cli ping > /dev/null 2>&1; then
    echo "✅ Redis is running"
else
    echo "❌ Redis failed to start"
    exit 1
fi

# Start Celery worker in background
echo "👷 Starting Celery worker..."
celery -A celery_app worker --loglevel=info --logfile=celery_worker.log --detach

# Start Celery flower for monitoring (optional)
echo "🌸 Starting Celery Flower monitoring..."
pip install flower 2>/dev/null
celery -A celery_app flower --port=5555 --detach

echo "✅ All services started!"
echo ""
echo "📊 Monitor at:"
echo "  - Celery Flower: http://localhost:5555"
echo "  - Agent Monitor: http://localhost:8000/static/monitor.html"
echo ""
echo "To stop services:"
echo "  redis-cli shutdown"
echo "  pkill -f celery"