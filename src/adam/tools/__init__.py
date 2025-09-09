"""
ADAM Tools Package
External capabilities for enhanced functionality
"""

from .sql_tools import SQLAnalyzer, SQLFormatter, SQLOptimizer

# New tool imports (will be added as we create them)
try:
    from .web_search import WebSearchTool
except ImportError:
    WebSearchTool = None

# Better approach - use model's native web search
try:
    from .model_web_search import ModelWebSearch, ADAMWebSearchIntegration
except ImportError:
    ModelWebSearch = None
    ADAMWebSearchIntegration = None

try:
    from .code_executor import CodeExecutor
except ImportError:
    CodeExecutor = None

try:
    from .file_generator import FileGenerator
except ImportError:
    FileGenerator = None

__all__ = [
    'SQLAnalyzer', 
    'SQLFormatter', 
    'SQLOptimizer',
    'WebSearchTool',
    'CodeExecutor',
    'FileGenerator'
]