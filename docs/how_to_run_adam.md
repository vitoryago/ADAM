# How to Run ADAM - Complete Guide

## Quick Start

### 1. Basic Requirements
```bash
# Make sure you're in the ADAM directory
cd ~/ADAM

# Ensure your virtual environment is activated
source venv/bin/activate

# Install optional dependencies for better experience
pip install rich  # For colored output (optional but recommended)
```

### 2. Check Your API Keys
```bash
# Verify API keys are set
python test_llm_setup.py

# If not set, add them to .env file
echo 'XAI_API_KEY=your-xai-key' >> .env
echo 'OPENAI_API_KEY=your-openai-key' >> .env
```

### 3. Run ADAM

You have three options:

#### Option A: Complete Interface (Recommended)
```bash
python adam_complete.py
```
This gives you:
- Full transparency (see which model is selected and why)
- Memory system tracking
- Cost tracking
- SQL analysis
- Complete visibility into ADAM's thinking

#### Option B: Simple Chat (Quick Testing)
```bash
python adam_simple_chat.py
```
Lightweight version with:
- Basic LLM chat
- SQL analysis
- No memory system

#### Option C: Test Mode
```bash
python adam_complete.py --test
```
Runs automated tests to verify everything works.

## Understanding ADAM's Interface

### What You'll See

When you run `python adam_complete.py`, you'll see:

```
╔════════════════════════════════════════════════════════════╗
║                  🧠 ADAM Complete Interface                ║
║              Analytics Data Assistant & Manager            ║
║                                                           ║
║        Full transparency mode - See everything!           ║
╚════════════════════════════════════════════════════════════╝

⏳ Initializing memory system...
⏳ Initializing conversation system...
⏳ Initializing memory network...
⏳ Initializing RAG system...
⏳ Initializing LLM client...

┌─────────────────────────────┐
│ Available LLM Models        │
├─────────────────────────────┤
│ Model         Provider  ... │
│ grok-4        grok          │
│ grok-3-mini   grok          │
│ gpt-4         openai        │
│ gpt-3.5-turbo openai        │
└─────────────────────────────┘

✅ ADAM is fully initialized and ready!
Type 'help' for commands or start chatting about analytics!

You: 
```

### How Model Selection Works

ADAM automatically selects the best model based on your query:

1. **SQL/Analytics Content** → `grok-4`
   - Keywords: sql, query, snowflake, optimize, join
   - Example: "How do I optimize this Snowflake query?"

2. **Reasoning/Debugging** → `o4-mini` or `grok-3-mini`
   - Keywords: explain, why, debug, understand
   - Example: "Why is my dbt model failing?"

3. **Simple Questions** → `gpt-3.5-turbo`
   - Short, straightforward queries
   - Example: "What is a CTE?"

4. **Complex Analysis** → `gpt-4` or `grok-4`
   - Long, detailed questions
   - Example: Complex business logic questions

### Available Commands

| Command | Description |
|---------|-------------|
| `help` | Show all available commands |
| `stats` | Display session statistics and costs |
| `models` | Show detailed model information |
| `memory` | Display memory system information |
| `exit` or `quit` | End the session |

## Example Interactions

### 1. Basic Analytics Question
```
You: What's the difference between a fact and dimension table?

🔍 Searching memory...
  Found 2 relevant memories
  [1] Score: 0.823 | Method: vector
  
🤖 Selecting best model...
  Auto-selected: grok-4
  Reason: Analytics content detected → grok-4 preferred

💭 Generating response with grok-4...

ADAM: A fact table contains measurable, quantitative data about business events...

[Model: grok-4 | Cost: $0.0021 | Time: 1.3s]
```

### 2. SQL Query Analysis
```
You: SELECT * FROM orders o, customers c WHERE o.customer_id = c.id

🔍 Detected SQL query - running analysis...

📊 SQL Query Analysis

Query Metrics:
- Complexity Score: 2/10
- Lines: 1
- Joins: 0
- Subqueries: 0

Issues Found (2):

⚠️ WARNINGS:
- Avoid SELECT *, specify needed columns
  → List specific columns to reduce data transfer
- Implicit cross join detected (comma-separated tables)
  → Use explicit JOIN syntax for clarity
```

### 3. Optimization Request
```
You: optimize

🚀 Optimizing query...

🎯 Optimized Query:

SELECT 
    o.order_id,
    o.order_date,
    o.amount,
    c.customer_name,
    c.email
FROM orders o
JOIN customers c ON o.customer_id = c.id

Estimated Improvement: Potential for 20-50% performance improvement
```

## Understanding the Output

### Model Selection Transparency
ADAM shows you:
- Which model was auto-selected
- Why that model was chosen
- The final model used (might differ if first choice fails)
- Token count and cost

### Memory System
When enabled, ADAM:
- Searches past conversations for relevant info
- Shows how many memories were found
- Displays retrieval scores and methods
- Stores valuable exchanges for future use

### Cost Tracking
- Every response shows the cost
- Use `stats` command to see total session cost
- Different models have different costs:
  - gpt-3.5-turbo: ~$0.001 per 1K tokens
  - gpt-4: ~$0.03 per 1K tokens
  - grok models: Variable pricing

## Advanced Usage

### Quiet Mode
For less verbose output:
```bash
python adam_complete.py --quiet
```

### Direct SQL Analysis
Just paste any SQL query directly:
```sql
WITH monthly_sales AS (
    SELECT 
        DATE_TRUNC('month', order_date) as month,
        SUM(amount) as total
    FROM orders
    GROUP BY 1
)
SELECT * FROM monthly_sales
```

### Memory Inspection
Use the `memory` command to see:
- Total memories stored
- Memory types distribution
- Recent memories
- Network connections

## Tips for Best Results

1. **Be Specific**: Instead of "optimize this", provide context
2. **Use Domain Terms**: Mention Snowflake, dbt, BigQuery for better model routing
3. **Paste Actual SQL**: ADAM analyzes real queries better than descriptions
4. **Ask Follow-ups**: ADAM remembers context within a session

## Troubleshooting

### API Key Issues
```bash
# Check if keys are loaded
python -c "import os; print('XAI:', 'Set' if os.getenv('XAI_API_KEY') else 'Not set')"

# Re-export if needed
export XAI_API_KEY="your-key"
export OPENAI_API_KEY="your-key"
```

### Module Import Errors
```bash
# Ensure you're in virtual environment
which python  # Should show ADAM/venv/bin/python

# Reinstall dependencies
pip install -r requirements.txt
```

### Memory/Performance Issues
```bash
# Clear old memory data
rm -rf adam_complete_memory/
rm -rf adam_complete_conversations/

# Run with fresh memory
python adam_complete.py
```

## Session Management

Your conversations are automatically saved. When you exit:
- Session is saved with timestamp
- Memories are persisted
- You can continue later with accumulated knowledge

## Next Steps

1. Try different types of queries to see model selection
2. Build up ADAM's memory with your specific use cases
3. Use SQL analysis for your actual queries
4. Check `stats` periodically to monitor costs

Remember: ADAM gets smarter with use. The more you interact, the better the memory system becomes at helping you!