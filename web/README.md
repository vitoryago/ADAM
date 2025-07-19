# ADAM Web Interface

Modern web-based interface for ADAM built with Streamlit.

## Available Interface

### adam_web.py
Clean, ChatGPT-style web interface for ADAM with:
- Session management and persistence
- Conversation history with date grouping
- Model selection (grok-4, grok-3-mini, GPT-4, etc.)
- Memory statistics and hit rate display
- Cost tracking per session
- Image upload support (grok-4 only)
- Toggle for memory search (performance optimization)
- Improved context handling
- Real-time streaming responses

**Usage:**
```bash
# From the ADAM root directory:
streamlit run web/adam_web.py

# Or with specific port:
streamlit run web/adam_web.py --server.port 8501
```

**Note:** You must use `streamlit run`, not `python`. Running with `python` will cause errors.

### demo_web_interface.py
Demo script showing the web interface capabilities.

## Features

The interface includes:
- 🧠 **Perfect Memory**: ADAM remembers all conversations
- 🤖 **Intelligent Routing**: Automatically selects the best model
- 💰 **Cost Tracking**: See how much each conversation costs
- 📊 **Analytics**: View memory usage and conversation statistics
- 🔄 **Session Management**: Continue previous conversations
- 🖼️ **Image Support**: Upload and analyze images (grok-4 only)

## Requirements
```bash
pip install -r requirements_web.txt
```

## Environment Setup
Create a `.env` file with:
```
XAI_API_KEY=your-grok-api-key
OPENAI_API_KEY=your-openai-api-key
```

## Quick Start
1. Install dependencies: `pip install -r requirements_web.txt`
2. Set up your API keys in `.env`
3. Run: `streamlit run web/adam_web.py`
4. Open browser to http://localhost:8501

## Customization
The interfaces can be customized by modifying:
- CSS styles in the markdown sections
- Page configuration in `st.set_page_config()`
- Model selection logic
- Memory display settings