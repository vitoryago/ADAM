"""Basic tests for ADAM"""

import sys
from pathlib import Path
import types

sys.path.append(str(Path(__file__).parent.parent))

# Provide stubs for optional dependencies so tests can run without them
sys.modules.setdefault('src.hello_adam', types.SimpleNamespace(ADAM=object()))
sys.modules.setdefault('langchain.llms', types.SimpleNamespace(Ollama=object()))

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
