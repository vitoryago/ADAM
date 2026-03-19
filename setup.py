#!/usr/bin/env python3
"""
Setup script for ADAM - Analytics Data Assistant with Memory
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read the README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name="adam-assistant",
    version="4.0.0",
    author="ADAM Development Team",
    author_email="adam@example.com",
    description="Analytics Data Assistant with Memory - An intelligent AI assistant with conversation tracking and memory networks",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/adam",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    python_requires=">=3.9",
    install_requires=[
        # Web Framework
        "fastapi>=0.104.0",
        "uvicorn[standard]>=0.24.0",
        "pydantic>=2.5.0",
        "python-multipart>=0.0.6",
        # Database
        "sqlalchemy>=2.0.23",
        "aiosqlite>=0.19.0",
        # LLM Providers
        "openai>=1.6.0",
        "anthropic>=0.8.0",
        "langchain>=0.1.0",
        "langchain-community>=0.0.13",
        "langchain-openai>=0.0.5",
        "langchain-anthropic>=0.1.0",
        "langgraph>=0.0.26",
        # Memory & ML
        "chromadb>=0.4.0",
        "sentence-transformers>=2.2.0",
        "networkx>=3.0",
        "numpy>=1.24.0",
        "scikit-learn>=1.3.0",
        # Utilities
        "python-dotenv>=1.0.0",
        "pyyaml>=6.0",
        "ruamel.yaml>=0.17.0",
        "httpx>=0.25.0",
        "aiohttp>=3.9.0",
        "websockets>=12.0",
        # SQL/dbt
        "sqlparse>=0.4.4",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-asyncio>=0.23.0",
            "pytest-cov>=4.1.0",
            "black>=23.0.0",
            "ruff>=0.1.0",
            "mypy>=1.0.0",
        ],
        "voice": [
            "elevenlabs>=0.2.27",
            "pyttsx3>=2.90",
            "SpeechRecognition>=3.10.0",
        ],
        "viz": [
            "matplotlib>=3.7.0",
            "seaborn>=0.12.0",
            "plotly>=5.18.0",
            "pydot>=1.4.2",
        ],
    },
    entry_points={
        "console_scripts": [
            "adam-server=uvicorn:main",
        ],
    },
)
