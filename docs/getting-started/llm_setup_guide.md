# LLM Setup Guide for ADAM

## Quick Start (2 minutes)

### Step 1: Set Your API Keys

You have three options for setting API keys:

#### Option 1: Environment Variables (Recommended for Testing)
```bash
# In your terminal, set the keys for this session:
export XAI_API_KEY="your-xai-api-key-here"
export OPENAI_API_KEY="your-openai-api-key-here"
```

#### Option 2: .env File (Recommended for Development)
```bash
# Copy the example file
cp .env.example .env

# Edit .env and add your keys:
# XAI_API_KEY=your-actual-xai-key
# OPENAI_API_KEY=your-actual-openai-key
```

#### Option 3: Shell Profile (Permanent)
```bash
# Add to ~/.bashrc or ~/.zshrc:
echo 'export XAI_API_KEY="your-xai-api-key-here"' >> ~/.zshrc
echo 'export OPENAI_API_KEY="your-openai-api-key-here"' >> ~/.zshrc
source ~/.zshrc
```

### Step 2: Test Your Setup
```bash
cd ~/ADAM
python test_llm_setup.py
```

## API Key Information

### For Grok Models (grok-4, grok-3-mini)
- **Get API Key**: https://x.ai/api
- **Models Available**:
  - `grok-4`: Most capable, best for complex analysis
  - `grok-3-mini`: Fast reasoning model with `reasoning_effort` parameter

### For OpenAI Models (o4-mini-high)
- **Get API Key**: https://platform.openai.com/api-keys
- **Models Available**:
  - `o4-mini-high`: Advanced reasoning with `effort` parameter

## Using the LLM System

### Basic Usage
```python
from adam.llm.client import UnifiedLLMClient

# Initialize client (auto-loads from environment)
client = UnifiedLLMClient()

# Let ADAM choose the best model
response = await client.complete("Explain why SQL queries can be slow")
print(response.content)

# Use a specific model
response = await client.complete(
    "Debug this dbt error: 'relation does not exist'",
    model="grok-4"
)

# Use reasoning models
response = await client.complete(
    "Why is my incremental model doing full refreshes?",
    model="o4-mini-high",
    reasoning_effort="high"
)
```

### Model Selection Guide

ADAM automatically selects models based on your query:

1. **SQL/Analytics Keywords** → `grok-4`
   - "optimize query", "SQL", "dbt", "Snowflake"
   
2. **Reasoning/Analysis** → `o4-mini-high`
   - "analyze", "explain", "debug", "why"
   
3. **Fast Responses** → `grok-3-mini`
   - Simple questions, quick lookups

### Cost Tracking

Each response includes cost information:
```python
response = await client.complete(query)
print(f"Cost: ${response.cost:.4f}")
print(f"Tokens: {response.total_tokens}")
```

## Troubleshooting

### No API Keys Found
```
❌ No API keys found!
To use ADAM's LLM capabilities, you need to set up your API keys:

1. For Grok models (grok-4, grok-3-mini):
   export XAI_API_KEY="your-xai-api-key-here"
   
2. For OpenAI models (o4-mini):
   export OPENAI_API_KEY="your-openai-api-key-here"
```

### Testing Individual Models
```bash
# Test only Grok models
export XAI_API_KEY="your-key"
python test_llm_setup.py

# Test only OpenAI models  
export OPENAI_API_KEY="your-key"
python test_llm_setup.py
```

### Common Issues

1. **Import Errors**: Install required packages
   ```bash
   pip install xai-sdk openai
   ```

2. **API Errors**: Check your API key is valid and has credits

3. **Model Not Available**: Ensure you have access to the specific model

## Next Steps

Once your LLM setup is working:

1. **Test with Real Queries**:
   ```python
   # Analytics question
   await quick_complete("How do I optimize a Snowflake query with window functions?")
   
   # Debugging help
   await reasoning_complete("Why is my dbt incremental model slow?", effort="high")
   ```

2. **Integrate with ADAM's Memory**:
   - Expensive responses are automatically stored
   - Reasoning chains are preserved
   - Knowledge builds over time

3. **Start the Roadmap Implementation**:
   - Week 1: LLM Integration ✅
   - Week 1: SQL Tools (next)
   - Week 2: dbt Integration
   - Week 3: Advanced Features

Your LLM system is now ready! Run `python test_llm_setup.py` to verify everything works.