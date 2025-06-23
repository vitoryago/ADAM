"""Basic tests for ADAM"""

import sys
from pathlib import Path
import pytest

sys.path.append(str(Path(__file__).parent.parent))

def test_adam_imports():
    """Ensure ADAM module can be imported if present."""
    pytest.importorskip("src.hello_adam")

def test_ollama_available():
    """Ensure Ollama can be imported if dependencies are installed."""
    pytest.importorskip("langchain.llms")
