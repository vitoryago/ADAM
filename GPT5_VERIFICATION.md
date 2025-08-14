# GPT-5 Organization Verification Required

## Current Status
OpenAI requires organization verification to use GPT-5 models with streaming.

## Error Message
```
Your organization must be verified to stream this model. 
Please go to: https://platform.openai.com/settings/organization/general 
and click on Verify Organization.
```

## Verification Steps

1. **Go to OpenAI Platform Settings**:
   https://platform.openai.com/settings/organization/general

2. **Click "Verify Organization"**

3. **Wait 15 minutes** for access to propagate

## Temporary Workaround

I've added a temporary fix that disables streaming for GPT-5 models until verification is complete. This means:
- ✅ GPT-5 will work (non-streaming)
- ⚠️ Responses will appear all at once instead of streaming
- ✅ Full functionality will be restored after verification

## After Verification

Once your organization is verified:

1. Remove the temporary workaround in `src/adam/llm/client.py` (lines 410-415)
2. Restart the backend
3. GPT-5 will work with full streaming support

## Alternative Models (No Verification Needed)

While waiting for verification, you can use:
- **Grok models**: grok-4, grok-3-mini-high (with your XAI_API_KEY)
- **Automatic routing**: Will use available models intelligently

## Testing After Verification

Test streaming is working:
```python
# Quick test
curl -X POST http://localhost:8000/api/messages/stream \
  -H "Content-Type: application/json" \
  -d '{"content": "Hello", "model": "gpt-5-mini"}'
```

If streaming works, you'll see chunks appear progressively.