# ADAM Web Interface Performance Tips

## Issue: Slow Response Times

If ADAM is taking too long to respond, here are some optimizations:

### 1. **Model Selection**
- **Avoid grok-4-reasoning** for casual chat - it's designed for complex reasoning tasks
- Use **grok-3-mini-high** for quick conversations
- Use **grok-4** only when you need advanced capabilities

### 2. **Memory Search Settings**
- **Disable "Search long-term memory"** in the sidebar for faster responses
- This prevents ADAM from searching through all past conversations
- Enable it only when you need to reference past discussions

### 3. **Conversation Context**
The web interface has been updated to:
- Prioritize current conversation context over memory search
- Include only the last 3 exchanges for context (not the entire history)
- Only search memory when explicitly enabled

## Quick Settings for Fast Chat

1. In the sidebar under "Settings":
   - Select Model: **grok-3-mini-high**
   - Uncheck: **Search long-term memory**

2. This gives you:
   - Fast response times (usually < 2 seconds)
   - Good conversation continuity
   - Lower costs

## When to Use Each Model

### grok-3-mini-high (Fastest)
- Casual conversation
- Simple questions
- Quick lookups
- Memory recaps

### grok-4 (Balanced)
- Technical questions
- Code analysis
- Detailed explanations
- Complex queries

### grok-4-reasoning (Slowest)
- Deep analysis
- Complex code generation
- System design
- Multi-step reasoning

## Technical Details

The updated `adam_web.py` now:
1. Maintains conversation context from current session
2. Only searches long-term memory when enabled
3. Limits context to prevent token overflow
4. Uses streaming for perceived faster responses