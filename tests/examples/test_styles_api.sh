#!/bin/bash
# Test Response Styles via API

echo "Testing ADAM Response Styles"
echo "============================"

# Get project ID (using the first project or create one)
PROJECT_ID=$(curl -s http://localhost:8000/api/projects | python -c "import sys, json; data = json.load(sys.stdin); print(data[0]['id'] if data else '')")

if [ -z "$PROJECT_ID" ]; then
    echo "Creating test project..."
    PROJECT_ID=$(curl -s -X POST http://localhost:8000/api/projects \
        -H "Content-Type: application/json" \
        -d '{"name": "Style Test", "description": "Testing response styles"}' | \
        python -c "import sys, json; print(json.load(sys.stdin)['id'])")
fi

echo "Using Project ID: $PROJECT_ID"
echo

# Create conversation
echo "Creating conversation..."
CONV_ID=$(curl -s -X POST "http://localhost:8000/api/projects/$PROJECT_ID/conversations" \
    -H "Content-Type: application/json" \
    -d '{"title": "Style Test Conversation"}' | \
    python -c "import sys, json; print(json.load(sys.stdin)['id'])")

echo "Conversation ID: $CONV_ID"
echo

# Test each style
STYLES=("concise" "normal" "explanatory" "friendly")
QUERY="What is artificial intelligence?"

for STYLE in "${STYLES[@]}"; do
    echo "----------------------------------------"
    echo "Testing $STYLE style:"
    echo "----------------------------------------"
    
    # Send message with specific style
    RESPONSE=$(curl -s -X POST "http://localhost:8000/api/conversations/$CONV_ID/messages" \
        -H "Content-Type: application/json" \
        -d "{
            \"content\": \"$QUERY\",
            \"use_memory\": false,
            \"model\": \"gpt-4.1-mini-2025-04-14\",
            \"response_style\": \"$STYLE\"
        }")
    
    # Extract and display assistant response
    echo "$RESPONSE" | python -c "
import sys, json
try:
    data = json.load(sys.stdin)
    if isinstance(data, list):
        for msg in data:
            if msg.get('role') == 'assistant':
                content = msg.get('content', 'No content')
                words = len(content.split())
                print(f'Response ({words} words):')
                print(content[:500] + ('...' if len(content) > 500 else ''))
                break
except Exception as e:
    print(f'Error parsing response: {e}')
    print('Raw response:', sys.stdin.read())
    "
    echo
done

echo "============================"
echo "Test Complete!"