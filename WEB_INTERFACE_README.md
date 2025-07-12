# ADAM Web Interface

Welcome to ADAM's web interface! This provides a modern, ChatGPT-like experience for interacting with ADAM, complete with image support, session management, and memory visualization.

## 🚀 Quick Start

1. **Install dependencies:**
   ```bash
   pip install streamlit
   ```

2. **Set your API keys:**
   ```bash
   export XAI_API_KEY="your-xai-key"    # For Grok models (required for images)
   export OPENAI_API_KEY="your-openai-key"  # For OpenAI models (optional)
   ```

3. **Launch the interface:**
   ```bash
   # Basic interface
   streamlit run adam_web.py
   
   # Or advanced interface with enhanced UI
   streamlit run adam_web_advanced.py
   ```

4. **Open your browser:**
   Navigate to http://localhost:8501

## 🎯 Features

### Core Features (Both Interfaces)
- **💬 Chat Interface**: Natural conversation with ADAM
- **🖼️ Image Analysis**: Upload images for analysis (grok-4 only)
- **📝 Session Management**: Create and manage multiple conversations
- **🕒 Conversation History**: Access all past conversations
- **🧠 Memory Context**: See relevant memories ADAM uses
- **💰 Cost Tracking**: Monitor API usage costs
- **🤖 Model Selection**: Choose between available models

### Advanced Interface Extras
- **✨ Modern UI**: Gradient styling and enhanced visuals
- **🔄 Real-time Streaming**: See responses as they're generated
- **🔍 Session Search**: Find conversations quickly
- **⚙️ Advanced Settings**: Toggle memory display and streaming
- **📊 Enhanced Analytics**: Better memory stats display

## 📸 Image Support

To use image analysis:
1. Select "grok-4" model in the sidebar settings
2. Upload an image using the 📸 button or file uploader
3. Add your question about the image
4. Send your message

**Supported formats**: PNG, JPG, JPEG, GIF, WEBP

## 🧠 Understanding Memory Context

When ADAM responds, it may show "Used memory context" - this means:
- ADAM found relevant information from past conversations
- The response is informed by previous interactions
- You're benefiting from ADAM's perfect memory system

## 💡 Tips for Best Experience

1. **Start Fresh**: Click "✨ New Conversation" for new topics
2. **Use Descriptive Titles**: First message becomes the session title
3. **Model Selection**:
   - `grok-4`: Best for complex tasks and image analysis
   - `grok-3-mini`: Fast responses for simple queries
   - `o4-mini-high`: High reasoning capability
   - `gpt-4`: OpenAI's most capable model
   - `gpt-3.5-turbo`: Fast and cost-effective

4. **Memory Building**: The more you use ADAM, the better it remembers

## 🔧 Troubleshooting

### Server won't start
- Check API keys are set correctly
- Ensure port 8501 is not in use
- Try: `lsof -i :8501` to check port usage

### Images not working
- Ensure you've selected "grok-4" model
- Check image file size (keep under 10MB)
- Verify XAI_API_KEY is set

### Memory not showing
- Memory builds over time with usage
- Check "Show memory context" is enabled (advanced interface)
- Ensure memories exist for your topic

## 🛠️ Configuration

### Environment Variables
```bash
XAI_API_KEY=your-key         # Required for Grok models
OPENAI_API_KEY=your-key      # Optional for OpenAI models
```

### Custom Port
```bash
streamlit run adam_web.py --server.port 8502
```

### Network Access
```bash
streamlit run adam_web.py --server.address 0.0.0.0
```

## 📊 Understanding the Interface

### Sidebar Components
- **ADAM Logo & Title**: Click to return home
- **Metrics**: Total memories, hit rate, session cost
- **New Conversation**: Start fresh dialogue
- **Search**: Find past conversations
- **Recent Conversations**: Grouped by date
- **Settings**: Model selection and preferences

### Main Chat Area
- **Session Title**: Current conversation name
- **Message History**: Full conversation thread
- **Metadata**: Model used, cost, memory usage
- **Input Area**: Type messages and upload images

## 🎨 Interface Comparison

| Feature | Basic (`adam_web.py`) | Advanced (`adam_web_advanced.py`) |
|---------|----------------------|-----------------------------------|
| Core Functionality | ✅ | ✅ |
| Image Support | ✅ | ✅ |
| Modern UI | Basic | Enhanced with gradients |
| Response Streaming | ✅ | ✅ with toggle |
| Session Search | Basic list | Advanced with search |
| Memory Toggle | No | Yes |
| Visual Polish | Standard | Premium |

## 🚀 Next Steps

1. **Explore**: Try different types of queries
2. **Build Memory**: Regular use improves ADAM's recall
3. **Test Images**: Upload screenshots or diagrams
4. **Compare Models**: Try same query with different models
5. **Track Costs**: Monitor usage in the sidebar

## 📝 Example Interactions

### Code Help
```
You: "Help me optimize this Python function for speed"
[Upload code screenshot or paste code]
ADAM: [Provides optimization suggestions with memory context]
```

### Image Analysis
```
You: "What's wrong with this error message?"
[Upload error screenshot]
ADAM: [Analyzes image and provides solution]
```

### Project Planning
```
You: "Let's design a REST API for a todo app"
ADAM: [Uses memory of past API discussions to provide tailored advice]
```

---

Enjoy your enhanced ADAM experience! The web interface brings all of ADAM's capabilities into a modern, user-friendly format. 🚀