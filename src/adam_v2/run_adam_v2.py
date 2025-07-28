#!/usr/bin/env python3
"""
Run ADAM v2.0 with proper environment setup
"""
import sys
import os
import subprocess

print("🚀 Starting ADAM v2.0...")
print(f"Python: {sys.executable}")
print(f"Version: {sys.version}")
print(f"Working directory: {os.getcwd()}")

# Check if we're in the right directory
if not os.path.exists("main.py"):
    print("❌ Error: main.py not found. Please run from adam_v2 directory")
    sys.exit(1)

# Check dependencies
print("\n📦 Checking dependencies...")
try:
    import fastapi
    print("✅ FastAPI installed")
except ImportError:
    print("❌ FastAPI not installed")
    
try:
    import uvicorn
    print("✅ Uvicorn installed")
except ImportError:
    print("❌ Uvicorn not installed")

try:
    import chromadb
    print("✅ ChromaDB installed (memory system available)")
except ImportError:
    print("⚠️  ChromaDB not installed (memory system disabled)")

# Run the server
print("\n🌐 Starting server on http://localhost:8000")
print("📝 Note: Memory features are disabled without ChromaDB")
print("Press Ctrl+C to stop\n")

# Run with subprocess to see output
subprocess.run([sys.executable, "main.py"])