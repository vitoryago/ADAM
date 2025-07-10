#!/bin/bash

# ADAM Setup Script - Designed for your GitHub repository
# This script creates ADAM's structure inside your existing ADM repo

echo "🚀 Setting up ADAM in your GitHub repository"
echo "==========================================="
echo ""

# First, let's check if we're in a git repository
if [ ! -d .git ]; then
    echo "⚠️  Warning: This doesn't appear to be a git repository."
    echo "Make sure you're in your ADM folder!"
    exit 1
fi

echo "✅ Found git repository: $(basename $(pwd))"
echo ""

# Create the project structure
echo "📁 Creating ADAM's directory structure..."

# Core directories
mkdir -p src/adam/{core,memory,vision,voice,tools}
mkdir -p docs/{architecture,tutorials,costs,daily_logs}
mkdir -p knowledge/{schemas,patterns,business_context,languages}
mkdir -p tests/{unit,integration}
mkdir -p config
mkdir -p notebooks  # For Jupyter experiments
mkdir -p scripts   # For utility scripts

# Create README if it doesn't exist
if [ ! -f README.md ]; then
    cat > README.md << 'EOF'
# ADAM - Advanced Data Analytics Model

An AI-powered analytics assistant that helps data engineers work faster and smarter.

## Overview

ADAM is a voice-enabled AI assistant specifically designed for analytics engineers. It can:
- Generate and explain SQL queries from natural language
- Understand your screen and provide context-aware help
- Assist with dbt model development
- Help prepare for meetings with data insights
- Communicate in English, Portuguese, and Spanish

## Project Structure

```
ADM/
├── src/adam/          # ADAM's core intelligence
├── knowledge/         # Domain knowledge and schemas
├── docs/             # Documentation and learning notes
├── notebooks/        # Jupyter notebooks for experiments
├── tests/            # Test suite
└── config/           # Configuration files
```

## Getting Started

1. Run the setup script: `./scripts/setup_environment.sh`
2. Activate the environment: `source venv/bin/activate`
3. Start ADAM: `python src/hello_adam.py`

## Development Log

Track daily progress in `docs/daily_logs/`

---
Built with ❤️ by an Analytics Engineer, for Analytics Engineers
EOF
fi

# Create .gitignore if it doesn't exist
if [ ! -f .gitignore ]; then
    cat > .gitignore << 'EOF'
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
venv/
env/
ENV/

# IDEs
.vscode/
.idea/
*.swp
*.swo
*~

# Project specific
.env
*.log
logs/
*.db
*.sqlite
chroma_db/
knowledge/local/
knowledge/private/

# ML Models (downloaded locally)
models/
*.bin
*.gguf

# OS
.DS_Store
Thumbs.db
Desktop.ini

# Jupyter
.ipynb_checkpoints/
*.ipynb_checkpoints

# Testing
.pytest_cache/
.coverage
htmlcov/
EOF
fi

# Create requirements.txt
cat > requirements.txt << 'EOF'
# Core AI/ML
langchain==0.1.1
langchain-community==0.0.13
ollama==0.1.4
chromadb==0.4.22
openai==1.6.1  # For vision capabilities when needed

# Voice capabilities
openai-whisper==20231117
pyttsx3==2.90
SpeechRecognition==3.10.1
pyaudio==0.2.14

# Data tools
pandas==2.1.4
sqlalchemy==2.0.25
dbt-core==1.7.4
sqlparse==0.4.4
psycopg2-binary==2.9.9  # PostgreSQL

# Utilities
python-dotenv==1.0.0
pydantic==2.5.3
pyyaml==6.0.1
rich==13.7.0  # Beautiful terminal output
click==8.1.7  # CLI tools

# Web interface
streamlit==1.29.0
gradio==4.13.0  # Alternative to Streamlit

# Development
pytest==7.4.4
black==23.12.1
ruff==0.1.9
ipykernel==6.28.0  # For Jupyter notebooks

# Documentation
mkdocs==1.5.3
mkdocs-material==9.5.3
EOF

# Create environment setup script
cat > scripts/setup_environment.sh << 'EOF'
#!/bin/bash

echo "🐍 Setting up ADAM's Python environment..."
echo ""

# Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Found Python: $python_version"

# Create virtual environment
echo "Creating virtual environment..."
python3 -m venv venv

# Activate it
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip wheel setuptools

# Install PyAudio dependencies (macOS)
if [[ "$OSTYPE" == "darwin"* ]]; then
    echo "📦 Installing audio dependencies for macOS..."
    if ! command -v brew &> /dev/null; then
        echo "⚠️  Homebrew not found. Please install it first."
        echo "Visit: https://brew.sh"
        exit 1
    fi
    brew install portaudio
fi

# Install Python packages
echo "📚 Installing Python packages..."
pip install -r requirements.txt

# Install Ollama if not present
if ! command -v ollama &> /dev/null; then
    echo "🤖 Installing Ollama..."
    if [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS installation
        curl -fsSL https://ollama.ai/install.sh | sh
    else
        echo "Please install Ollama manually from: https://ollama.ai"
    fi
fi

# Pull AI models
echo "🧠 Downloading AI models..."
echo "This might take a few minutes..."
ollama pull mistral
ollama pull llama2  # Backup model

# Create .env file from template
if [ ! -f .env ]; then
    echo "Creating .env file..."
    cp config/.env.template .env
    echo "⚠️  Please edit .env with your configuration"
fi

echo ""
echo "✅ Setup complete!"
echo ""
echo "Next steps:"
echo "1. Activate environment: source venv/bin/activate"
echo "2. Configure settings: edit .env"
echo "3. Test ADAM: python src/hello_adam.py"
EOF

chmod +x scripts/setup_environment.sh

# Create .env template
mkdir -p config
cat > config/.env.template << 'EOF'
# ADAM Configuration
ADAM_NAME=ADAM
ADAM_LANGUAGE=en  # Default language (en, pt, es)
ADAM_VOICE_SPEED=180
ADAM_VOICE_ENGINE=pyttsx3  # or 'elevenlabs' for premium

# Optional API Keys
OPENAI_API_KEY=your_key_here  # For GPT-4 Vision
ANTHROPIC_API_KEY=your_key_here  # For Claude
ELEVENLABS_API_KEY=your_key_here  # For premium voices

# Database Connections (Examples)
# SNOWFLAKE_ACCOUNT=
# SNOWFLAKE_USER=
# SNOWFLAKE_PASSWORD=
# POSTGRES_CONNECTION_STRING=

# Paths
DBT_PROJECT_PATH=
KNOWLEDGE_BASE_PATH=./knowledge
LOG_LEVEL=INFO
EOF

# Create the main ADAM file
cat > src/hello_adam.py << 'EOF'
#!/usr/bin/env python3
"""
ADAM - Advanced Data Analytics Model
Your AI-powered analytics engineering assistant
"""

import os
import sys
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from langchain.llms import Ollama
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
import pyttsx3
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.prompt import Prompt
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

console = Console()

class ADAM:
    """ADAM - Your Advanced Data Analytics Model assistant"""
    
    def __init__(self):
        console.print("[yellow]Initializing ADAM's neural pathways...[/yellow]")
        
        # Initialize the brain (LLM)
        self.llm = Ollama(
            model="mistral",
            callbacks=[StreamingStdOutCallbackHandler()],
            temperature=0.7
        )
        
        # Initialize voice
        self.voice_engine = pyttsx3.init()
        self.voice_engine.setProperty('rate', int(os.getenv('ADAM_VOICE_SPEED', 180)))
        
        # Set personality
        self.personality = """You are ADAM (Advanced Data Analytics Model), an AI assistant 
        specifically designed to help analytics engineers. You're knowledgeable about SQL, 
        dbt, data modeling, and analytics best practices. You speak clearly and helpfully,
        always aiming to teach and improve the user's skills."""
        
        console.print("[green]✅ ADAM is ready![/green]")
    
    def speak(self, text):
        """Convert text to speech"""
        self.voice_engine.say(text)
        self.voice_engine.runAndWait()
    
    def think(self, prompt):
        """Process a thought with full context"""
        full_prompt = f"{self.personality}\n\nUser: {prompt}\n\nADAM:"
        return self.llm.invoke(full_prompt)
    
    def introduce(self):
        """ADAM's introduction"""
        intro = """
# 🤖 ADAM - Advanced Data Analytics Model

Hello! I'm ADAM, your AI-powered analytics assistant.

I'm here to help you:
- **Write better SQL** - From simple queries to complex analytics
- **Master dbt** - Model development, testing, and best practices  
- **Debug faster** - Understand errors and find solutions
- **Learn continuously** - Improve your skills with every interaction

I can communicate in English, Portuguese, and Spanish. Just speak naturally!

Let's build amazing data solutions together! 🚀
        """
        
        console.print(Panel(Markdown(intro), title="Welcome", border_style="green"))
        
        # Spoken introduction
        self.speak(
            "Hello! I'm ADAM, your Advanced Data Analytics Model. "
            "I'm excited to help you become a more effective analytics engineer. "
            "Ask me anything about SQL, data modeling, or analytics!"
        )
    
    def chat(self):
        """Interactive chat session"""
        console.print("\n[green]Let's chat! Type 'exit' to end our session.[/green]\n")
        
        while True:
            # Get user input
            user_input = Prompt.ask("\n[blue]You[/blue]")
            
            if user_input.lower() in ['exit', 'quit', 'bye']:
                farewell = "Great chatting with you! Remember, every query you write makes you a better engineer. See you soon!"
                console.print(f"\n[green]ADAM:[/green] {farewell}")
                self.speak(farewell)
                break
            
            # Process and respond
            console.print("\n[green]ADAM:[/green] ", end="")
            self.think(user_input)
            print()  # New line after streaming
    
    def demo_sql_help(self):
        """Demonstrate SQL assistance capabilities"""
        console.print("\n[yellow]SQL Assistance Demo[/yellow]")
        demo_prompt = """
        The user needs help writing a SQL query to find customers who made 
        purchases in the last 30 days but not in the previous 30 days (new reactivated customers).
        Provide the SQL and explain the logic.
        """
        
        console.print("\n[green]ADAM:[/green] ", end="")
        self.think(demo_prompt)
        print()

def main():
    """Main entry point"""
    adam = ADAM()
    adam.introduce()
    
    # Optional: Show a demo
    if len(sys.argv) > 1 and sys.argv[1] == "--demo":
        adam.demo_sql_help()
    
    adam.chat()

if __name__ == "__main__":
    main()
EOF

# Create a simple test file
cat > tests/test_adam_basic.py << 'EOF'
"""Basic tests for ADAM"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

def test_adam_imports():
    """Test that ADAM can be imported"""
    try:
        from src.hello_adam import ADAM
        assert True
    except ImportError:
        assert False, "Failed to import ADAM"

def test_ollama_available():
    """Test that Ollama is available"""
    try:
        from langchain.llms import Ollama
        assert True
    except ImportError:
        assert False, "Ollama not available"
EOF

# Create first notebook for experiments
cat > notebooks/01_getting_started.ipynb << 'EOF'
{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# ADAM Development Notebook\n",
    "Experiment with ADAM's capabilities here!"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Test ADAM's components\n",
    "from langchain.llms import Ollama\n",
    "\n",
    "llm = Ollama(model=\"mistral\")\n",
    "response = llm.invoke(\"Explain CTEs in SQL\")\n",
    "print(response)"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
EOF

# Create today's log
mkdir -p docs/daily_logs
cat > docs/daily_logs/day_001.md << 'EOF'
# Day 1: ADAM's Birth

## Date: $(date +%Y-%m-%d)

### What I Built Today
- Set up ADAM's basic structure
- Got voice synthesis working
- Created first conversation interface
- Established project organization

### Key Learnings
- Ollama makes running LLMs locally surprisingly easy
- The M4 Pro handles Mistral model effortlessly
- Voice feedback creates more engaging interaction

### Challenges Faced
- [Document any setup issues here]

### Tomorrow's Goals
- Add memory to ADAM using ChromaDB
- Feed him first SQL patterns
- Create knowledge base structure

### Ideas for Future
- Meeting preparation assistant
- SQL query optimizer
- dbt model generator

### Code I'm Proud Of
[Paste any clever solutions here]

---
*"Every expert was once a beginner"*
EOF

# Create a quick start guide
cat > docs/QUICKSTART.md << 'EOF'
# ADAM Quick Start Guide

## First Time Setup (10 minutes)

1. **Install Dependencies**
   ```bash
   ./scripts/setup_environment.sh
   ```

2. **Activate Environment**
   ```bash
   source venv/bin/activate
   ```

3. **Configure ADAM**
   ```bash
   cp config/.env.template .env
   # Edit .env with your preferences
   ```

4. **Meet ADAM**
   ```bash
   python src/hello_adam.py
   ```

## Daily Development Flow

1. **Start your day**
   ```bash
   source venv/bin/activate
   git pull  # Get latest changes
   ```

2. **Work on ADAM**
   - Experiment in `notebooks/`
   - Build features in `src/adam/`
   - Document learnings in `docs/daily_logs/`

3. **End your day**
   ```bash
   git add .
   git commit -m "Day X: What I accomplished"
   git push
   ```

## Testing Ideas

Ask ADAM:
- "How do I write a CTE?"
- "Explain window functions"
- "Help me optimize this query"
- "What's wrong with my JOIN?"

## Troubleshooting

- **No audio?** Check microphone permissions
- **Slow responses?** Try the smaller llama2 model
- **Import errors?** Ensure venv is activated
EOF

echo ""
echo "✅ ADAM's home is ready in your GitHub repository!"
echo ""
echo "📋 Next steps:"
echo "1. Stage all files: git add ."
echo "2. Commit: git commit -m \"Initial ADAM setup\""
echo "3. Push: git push origin main"
echo "4. Run setup: ./scripts/setup_environment.sh"
echo ""
echo "You're about to bring ADAM to life! 🎉"