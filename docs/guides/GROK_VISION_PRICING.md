# Grok Vision Model Pricing

## Grok-2-Vision-1212 Pricing

As of the latest update, Grok-2 Vision model has the following pricing structure:

### Token Pricing
- **Input tokens**: $2.00 per million tokens
- **Output tokens**: $10.00 per million tokens
- **Cached input**: Same as regular input pricing

### Regional Pricing
- **us-east-1**: Same pricing as above
- **eu-west-1**: Same pricing as above

### Image Processing
Images are tokenized as part of the input tokens. Based on typical usage:
- Small images (~500x500): ~500-750 tokens
- Medium images (~1024x1024): ~1000-1500 tokens  
- Large images (~2048x2048): ~2000-3000 tokens

### Cost Examples
For a typical query with a medium-sized image:
- Image tokens: ~1000 tokens
- Text prompt: ~50 tokens
- Total input: ~1050 tokens
- Output response: ~200 tokens

**Cost calculation**:
- Input cost: (1050 / 1,000,000) × $2.00 = $0.0021
- Output cost: (200 / 1,000,000) × $10.00 = $0.0020
- **Total cost**: ~$0.0041 per query

### Comparison with Other Vision Models
- **GPT-4 Vision**: ~$0.01 per image (fixed) + text token costs
- **Grok-2 Vision**: Variable based on image size, typically $0.002-0.006 per image query
- **Cost advantage**: Grok-2 Vision is typically 50-80% cheaper than GPT-4V

### Implementation Details
The ADAM system now:
1. Tracks input and output tokens separately
2. Calculates costs based on the official pricing structure
3. Estimates image token usage based on image dimensions
4. Provides accurate cost tracking in the web interface