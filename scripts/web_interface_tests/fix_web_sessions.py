#!/usr/bin/env python3
"""
Fix or reset corrupted web sessions file
"""
import json
from pathlib import Path
import sys

def fix_sessions_file():
    """Fix or create new sessions file"""
    
    sessions_file = Path("data/web_sessions.json")
    corrupted_file = Path("data/web_sessions_corrupted.json.bak")
    
    # Ensure data directory exists
    sessions_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Try to salvage data from corrupted file
    salvaged_sessions = {}
    
    if corrupted_file.exists():
        print(f"Found corrupted file: {corrupted_file}")
        try:
            with open(corrupted_file, 'r') as f:
                content = f.read()
                
            # Try to parse what we can
            # The file seems to be truncated, so we'll try to extract valid JSON
            if content.strip():
                # Find the last complete brace/bracket
                # For now, just create a fresh file
                print("File is truncated. Creating fresh sessions file.")
        except Exception as e:
            print(f"Could not read corrupted file: {e}")
    
    # Create a new valid sessions file
    new_sessions = {}
    
    # If we want to preserve the session ID from the corrupted file
    if corrupted_file.exists():
        try:
            with open(corrupted_file, 'r') as f:
                lines = f.readlines()
                # Extract session ID from line 2 if possible
                for line in lines:
                    if "session_" in line and '"' in line:
                        import re
                        match = re.search(r'"(session_\d+_\d+_\w+)"', line)
                        if match:
                            session_id = match.group(1)
                            print(f"Found session ID: {session_id}")
                            # Create empty session
                            new_sessions[session_id] = {
                                "messages": [],
                                "total_cost": 0.0,
                                "selected_model": None,
                                "use_memory": True,
                                "last_updated": None
                            }
                            break
        except:
            pass
    
    # Write the new sessions file
    with open(sessions_file, 'w') as f:
        json.dump(new_sessions, f, indent=2)
    
    print(f"Created new sessions file: {sessions_file}")
    print(f"Sessions in file: {len(new_sessions)}")
    
    return True

if __name__ == "__main__":
    success = fix_sessions_file()
    sys.exit(0 if success else 1)