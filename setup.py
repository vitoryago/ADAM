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
    version="2.0.0",
    author="Your Name",
    author_email="your.email@example.com",
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
        "langchain>=0.1.0",
        "langchain-community>=0.0.13",
        "ollama>=0.1.0",
        "chromadb>=0.4.0",
        "sentence-transformers>=2.2.0",
        "networkx>=3.0",
        "numpy>=1.24.0",
        "scikit-learn>=1.3.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "rich>=13.0.0",
        "pyttsx3>=2.90",
        "python-dotenv>=1.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
        ],
        "vision": [
            "openai>=1.6.0",
            "openai-whisper>=20231117",
        ],
        "web": [
            "streamlit>=1.29.0",
            "gradio>=4.13.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "adam=adam_v2_memory:main",
        ],
    },
)