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
