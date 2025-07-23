#!/usr/bin/env python3
"""
Clean ADAM's Memory - Start Fresh
This script safely removes all stored memories and conversations
"""
import os
import shutil
from pathlib import Path

def clean_adam_memory():
    """Clean all ADAM memory data"""
    
    base_dir = Path(__file__).parent.parent
    
    # Directories to clean
    memory_dirs = [
        "adam_memory",
        "adam_memory_advanced",
        "adam_complete_memory",
        "adam_complete_conversations",
        "demo_conversations",
        "data/conversations",
        "data/memories"
    ]
    
    print("🧹 Cleaning ADAM's Memory...")
    print("=" * 50)
    
    for dir_name in memory_dirs:
        dir_path = base_dir / dir_name
        if dir_path.exists():
            print(f"\n📁 Cleaning: {dir_name}")
            
            # Special handling for ChromaDB directories
            if "adam_memory" in dir_name:
                # Remove ChromaDB data
                chroma_db = dir_path / "chroma.sqlite3"
                if chroma_db.exists():
                    print(f"  - Removing ChromaDB: {chroma_db.name}")
                    chroma_db.unlink()
                
                # Remove vector data directories
                for item in dir_path.iterdir():
                    if item.is_dir() and len(item.name) == 36:  # UUID directories
                        print(f"  - Removing vector data: {item.name}")
                        shutil.rmtree(item)
                
                # Remove metadata files
                for pattern in ["*.json", "*.log"]:
                    for file in dir_path.glob(pattern):
                        print(f"  - Removing: {file.name}")
                        file.unlink()
            
            # Clean conversation directories
            if "conversations" in str(dir_path):
                # Keep the directory but remove all session files
                for file in dir_path.glob("session_*.json"):
                    print(f"  - Removing session: {file.name}")
                    file.unlink()
            
            # Clean other directories
            if dir_path.exists() and any(dir_path.iterdir()):
                for item in dir_path.iterdir():
                    if item.is_file() and item.name != ".gitkeep":
                        print(f"  - Removing: {item.name}")
                        item.unlink()
    
    print("\n✅ ADAM's memory has been cleaned!")
    print("\nNext time you run ADAM, it will start with a fresh memory.")
    print("All your previous conversations and memories have been removed.")
    
    # Optional: Create a marker file to indicate fresh start
    marker = base_dir / "adam_memory_advanced" / ".fresh_start"
    marker.parent.mkdir(exist_ok=True)
    marker.touch()
    print("\n📝 Created fresh start marker")

if __name__ == "__main__":
    response = input("⚠️  This will permanently delete all of ADAM's memories and conversations.\nAre you sure you want to continue? (yes/no): ")
    
    if response.lower() in ['yes', 'y']:
        clean_adam_memory()
    else:
        print("❌ Cancelled. ADAM's memory remains intact.")