# Grok Vision Implementation Update

## Overview

Updated ADAM to properly support Grok vision models using the official xAI SDK format.

## Key Changes

### 1. Fixed Deprecation Warning
- Changed `use_column_width=True` to `use_container_width=True` in Streamlit image display

### 2. Proper xAI SDK Integration
- Import the `image` function from `xai_sdk.chat`
- Use the official format for image messages:
  ```python
  from xai_sdk.chat import user, image
  
  chat.append(
      user(
          "What's in this image?",
          image(image_url=f"data:image/jpeg;base64,{base64_image}", detail="high")
      )
  )
  ```

### 3. Image Token Calculation
Based on Grok documentation:
- Each image is broken into 448x448 pixel tiles
- Each tile consumes 256 tokens
- Maximum 6 tiles per image
- Formula: `(# of tiles + 1) * 256` tokens
- Typical image: ~1280 tokens (5 tiles)

### 4. Pricing Structure
For Grok vision models:
- Input: $2.00 per million tokens
- Output: $10.00 per million tokens
- Image tokens count as input tokens

## Supported Features

✅ Base64 encoded images
✅ Multiple images in one prompt
✅ High/Low/Auto detail levels
✅ Accurate token counting
✅ Cost calculation with image tokens

## Not Yet Implemented

❌ Web URL image input (requires fetching)
❌ Streaming with images

## Usage Example

```python
# In ADAM web interface:
1. Select a Grok vision model (marked with 🖼️)
2. Upload an image using the file uploader
3. Ask a question about the image
4. ADAM will analyze it using Grok vision

# Cost example for typical query:
- Text prompt: 50 tokens
- Image: 1280 tokens  
- Total input: 1330 tokens = $0.00266
- Output: 200 tokens = $0.002
- Total cost: ~$0.00466
```

## Technical Details

The implementation:
1. Checks if the model supports vision
2. Encodes uploaded images as base64
3. Formats the message using xAI SDK's `image()` function
4. Passes detail level as "high" for best quality
5. Tracks image token usage in the response