# GPT-5 Setup Guide for ADAM

## Quick Fix

ADAM now uses GPT-5 models, which require:
1. **OPENAI_API_KEY** environment variable
2. No custom temperature (GPT-5 only supports default temperature=1)
3. `max_completion_tokens` instead of `max_tokens`

## Setup Steps

### 1. Add OpenAI API Key

Add to your `.env` file in the ADAM root directory:
```
OPENAI_API_KEY=your-openai-api-key-here
```

Or export it in your terminal:
```bash
export OPENAI_API_KEY="your-openai-api-key-here"
```

### 2. Check Configuration

Run the diagnostic script:
```bash
python check_models.py
```

This will show:
- Which API keys are set
- Which models are available
- What needs to be configured

### 3. GPT-5 Model Limitations

GPT-5 models have specific requirements:
- **Temperature**: Only supports default value (1.0) - we handle this automatically
- **Max Tokens**: Uses `max_completion_tokens` instead of `max_tokens` - already fixed
- **Reasoning Effort**: Supports "minimal", "low", "medium", "high" levels

## Available Models

After setting up OPENAI_API_KEY, you'll have access to:
- **gpt-5**: Most capable, with vision support
- **gpt-5-mini**: Fast and efficient
- **gpt-5-nano**: Ultra-fast for simple queries

## Troubleshooting

### Models not showing in web interface
- Run `python check_models.py` to verify API keys
- Make sure OPENAI_API_KEY is set
- Restart the web interface after setting the key

### Getting parameter errors
The following issues have been fixed:
- ✅ `max_tokens` → `max_completion_tokens` for GPT-5
- ✅ Temperature parameter removed for GPT-5 (uses default)
- ✅ Reasoning effort properly mapped

### Still using Grok models?
If you see Grok models responding instead of GPT-5:
1. Make sure OPENAI_API_KEY is set
2. Select a GPT-5 model explicitly in the web interface
3. Or use "automatic" for smart routing

## Testing

After setup, test in the web interface:
1. Start ADAM web: `streamlit run web/adam_web.py`
2. You should see GPT-5 models in the dropdown
3. Send a test message like "Hey ADAM"
4. It should respond using GPT-5-mini (for simple queries)