#!/usr/bin/env python3
"""
ADAM Web Interface Demo Script
Shows how to launch and use the web interface
"""
import subprocess
import time
import webbrowser
import sys
import os
from pathlib import Path

def check_environment():
    """Check if environment is ready"""
    print("🔍 Checking environment...")
    
    # Check API keys
    has_xai = bool(os.getenv("XAI_API_KEY"))
    has_openai = bool(os.getenv("OPENAI_API_KEY"))
    
    if not (has_xai or has_openai):
        print("❌ No API keys found!")
        print("\nPlease set at least one of these environment variables:")
        print("  - XAI_API_KEY for Grok models (supports image analysis)")
        print("  - OPENAI_API_KEY for OpenAI models")
        return False
    
    print("✅ API keys found")
    if has_xai:
        print("   - XAI (Grok) models available ✅")
    if has_openai:
        print("   - OpenAI models available ✅")
    
    return True

def launch_interface():
    """Launch the web interface"""
    script_name = "adam_web.py"
    
    print(f"\n🚀 Launching ADAM Web Interface...")
    print(f"   Running: streamlit run {script_name}")
    
    # Start the streamlit server
    process = subprocess.Popen(
        ["streamlit", "run", script_name, "--server.headless", "true"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    # Wait for server to start
    print("   Waiting for server to start...")
    time.sleep(3)
    
    # Check if process is still running
    if process.poll() is not None:
        print("❌ Failed to start server!")
        stdout, stderr = process.communicate()
        print(f"Error: {stderr}")
        return None
    
    print("✅ Server started successfully!")
    return process

def show_usage_guide():
    """Show how to use the interface"""
    print("\n" + "="*60)
    print("📚 ADAM Web Interface Usage Guide")
    print("="*60)
    
    print("\n🎯 Features Available:")
    print("  1. 💬 Chat Interface - Talk to ADAM like ChatGPT/Claude")
    print("  2. 🖼️  Image Analysis - Upload images for grok-4 to analyze")
    print("  3. 📝 Session Management - Create and switch between conversations")
    print("  4. 🕒 Conversation History - Access all your past chats")
    print("  5. 🧠 Memory Context - See what ADAM remembers about topics")
    print("  6. 💰 Cost Tracking - Monitor API usage costs")
    print("  7. 🤖 Model Selection - Choose between available AI models")
    
    print("\n💡 Quick Start:")
    print("  1. Click '➕ New Conversation' in the sidebar to start")
    print("  2. Type your message in the chat input at the bottom")
    print("  3. (Optional) Upload an image using the 📸 button")
    print("  4. Press Enter to send your message")
    print("  5. ADAM will respond, showing memory context if relevant")
    
    print("\n🔧 Features:")
    print("  - Clean ChatGPT-style interface")
    print("  - Real-time response streaming")
    print("  - Session management")
    print("  - Memory context toggle for performance")
    print("  - Image upload support (grok-4 only)")
    
    print("\n⌨️  Keyboard Shortcuts:")
    print("  - Enter: Send message")
    print("  - Ctrl+Enter: New line in message")
    print("  - Esc: Clear input")

def main():
    """Main demo function"""
    print("🧠 ADAM Web Interface Demo")
    print("="*60)
    
    # Check environment
    if not check_environment():
        return
    
    # Launch the interface
    process = launch_interface()
    if not process:
        return
    
    # Show usage guide
    show_usage_guide()
    
    # Open browser
    print("\n🌐 Opening browser...")
    time.sleep(1)
    webbrowser.open("http://localhost:8501")
    
    print("\n✨ ADAM Web Interface is running!")
    print("   URL: http://localhost:8501")
    print("\n⏹️  Press Ctrl+C to stop the server...")
    
    try:
        # Keep running until user stops
        process.wait()
    except KeyboardInterrupt:
        print("\n\n🛑 Stopping server...")
        process.terminate()
        process.wait()
        print("✅ Server stopped successfully!")

if __name__ == "__main__":
    main()