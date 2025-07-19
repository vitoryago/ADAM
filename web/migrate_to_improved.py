#!/usr/bin/env python3
"""
Migration script to update adam_web.py with improvements from adam_web_improved.py
This script will backup the original and apply the improvements.
"""

import shutil
from pathlib import Path
import sys

def migrate_web_interface():
    """Migrate to improved web interface"""
    
    # Paths
    original = Path("adam_web.py")
    improved = Path("adam_web_improved.py")
    backup = Path("adam_web_original.py")
    
    if not original.exists():
        print("❌ Original adam_web.py not found!")
        return False
    
    if not improved.exists():
        print("❌ Improved version not found!")
        return False
    
    # Create backup
    print("📁 Creating backup of original...")
    shutil.copy2(original, backup)
    print(f"✅ Backup created: {backup}")
    
    # Copy improved version over original
    print("🔄 Applying improvements...")
    shutil.copy2(improved, original)
    print("✅ Improvements applied!")
    
    # Show what was added
    print("\n✨ Key improvements added:")
    print("  - Error boundaries for all operations")
    print("  - Session persistence to disk")
    print("  - System health indicators")
    print("  - Auto-save functionality")
    print("  - Better error handling and recovery")
    print("  - Toast notifications for saves")
    print("  - Improved UI/UX with loading states")
    
    print("\n🎉 Migration complete!")
    print("Run 'streamlit run adam_web.py' to see the improvements.")
    print(f"Original backed up to: {backup}")
    
    return True

if __name__ == "__main__":
    # Change to web directory if not already there
    web_dir = Path(__file__).parent
    import os
    os.chdir(web_dir)
    
    success = migrate_web_interface()
    sys.exit(0 if success else 1)