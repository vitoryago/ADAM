#!/usr/bin/env python3
"""
Setup script for ADAM Python Backend

This script helps users set up the ADAM Python backend by:
1. Creating necessary directories
2. Setting up a virtual environment (optional)
3. Installing dependencies
4. Configuring environment variables
5. Testing the installation

Usage:
    python setup.py --help
    python setup.py --install     # Install dependencies only
    python setup.py --full        # Full setup with venv
    python setup.py --test        # Test installation
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path

def create_directories():
    """Create necessary directories for ADAM"""
    dirs = [
        'adam_memory_advanced',
        'adam_memory_advanced/conversations',
        'adam_memory_advanced/cost_tracking',
        'logs',
        'temp'
    ]
    
    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        print(f"✓ Created directory: {dir_path}")

def create_env_template():
    """Create a .env template file"""
    # Don't overwrite if exists
    if Path('.env.template').exists():
        print("✓ .env.template already exists")
        return
    
    with open('.env.template', 'w') as f:
        f.write(env_template)
    
    print("✓ Created .env.template")
    print("  → Copy this to .env and add your API keys")

def install_dependencies(use_venv=False):
    """Install Python dependencies"""
    if use_venv:
        print("Creating virtual environment...")
        subprocess.run([sys.executable, '-m', 'venv', 'venv'])
        
        # Determine activation script path
        if os.name == 'nt':  # Windows
            pip_path = 'venv/Scripts/pip'
            python_path = 'venv/Scripts/python'
        else:  # Unix/MacOS
            pip_path = 'venv/bin/pip'
            python_path = 'venv/bin/python'
        
        print(f"Installing dependencies in virtual environment...")
        subprocess.run([pip_path, 'install', '-r', 'requirements.txt'])
        print("✓ Virtual environment created and dependencies installed")
        print(f"  → Activate with: source venv/bin/activate (Unix) or venv\\Scripts\\activate (Windows)")
    else:
        print("Installing dependencies globally...")
        subprocess.run([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'])
        print("✓ Dependencies installed globally")

def test_installation():
    """Test the ADAM installation"""
    print("Testing ADAM installation...")
    
    try:
        # Test core imports
        import numpy
        import chromadb
        import networkx
        import sentence_transformers
        print("✓ Core dependencies imported successfully")
        
        # Test LLM libraries
        try:
            import openai
            print("✓ OpenAI library available")
        except ImportError:
            print("⚠ OpenAI library not found (install if you want to use O1 models)")
        
        try:
            import anthropic
            print("✓ Anthropic library available")
        except ImportError:
            print("⚠ Anthropic library not found (install if you want to use Claude)")
            
        try:
            import xai_sdk
            print("✓ XAI SDK available")
        except ImportError:
            print("⚠ XAI SDK not found (install if you want to use Grok models)")
        
        # Test screen capture capabilities
        try:
            import mss
            print("✓ Screen capture (mss) available")
        except ImportError:
            print("⚠ MSS not found (install for screen capture: pip install mss)")
            
        try:
            import PIL
            print("✓ PIL/Pillow available")
        except ImportError:
            print("⚠ Pillow not found (install for image processing: pip install Pillow)")
        
        # Test ChromaDB initialization
        try:
            client = chromadb.Client()
            print("✓ ChromaDB client initialized")
        except Exception as e:
            print(f"⚠ ChromaDB initialization warning: {e}")
        
        print("\n✅ ADAM backend setup appears successful!")
        print("\nNext steps:")
        print("1. Copy .env.template to .env and add your API keys")
        print("2. Run: python main.py (to test the service)")
        print("3. Start your Node.js web application")
        
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("Run: python setup.py --install")
        return False
    
    return True

def main():
    parser = argparse.ArgumentParser(description='Setup ADAM Python Backend')
    parser.add_argument('--install', action='store_true', help='Install dependencies only')
    parser.add_argument('--full', action='store_true', help='Full setup with virtual environment')
    parser.add_argument('--test', action='store_true', help='Test installation')
    parser.add_argument('--venv', action='store_true', help='Use virtual environment')
    
    args = parser.parse_args()
    
    if not any(vars(args).values()):
        # No arguments provided, show help
        parser.print_help()
        return
    
    print("🤖 ADAM Python Backend Setup")
    print("=" * 40)
    
    if args.full or args.install:
        print("\n1. Creating directories...")
        create_directories()
        
        print("\n2. Creating environment template...")
        create_env_template()
        
        print("\n3. Installing dependencies...")
        install_dependencies(use_venv=args.full or args.venv)
    
    if args.test or args.full:
        print("\n4. Testing installation...")
        test_installation()

if __name__ == '__main__':
    main()