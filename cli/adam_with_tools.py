#!/usr/bin/env python3
"""
ADAM CLI with Tool Support - Simple synchronous version
"""

import sys
from pathlib import Path
import os
import re
import asyncio

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Tool imports
from adam.tools.file_tools import ReadFileTool, WriteFileTool, ListFilesTool
from adam.tools.system_tools import RunCommandTool


def process_message_with_tools(message: str):
    """Process a message and execute tools if needed"""
    
    message_lower = message.lower()
    
    # Check for file reading request
    if any(phrase in message_lower for phrase in ['read', 'show', 'look at', 'check', 'bring me a summary', 'summarize']):
        # Extract file path
        path_match = re.search(r'([~/\.]?[/\w\-_\.]+\.\w+)', message)
        if path_match:
            file_path = path_match.group(1)
            if file_path.startswith('~'):
                file_path = os.path.expanduser(file_path)
            
            print(f"\n🔧 Reading file: {file_path}")
            
            tool = ReadFileTool()
            # Run async tool in sync context
            result = asyncio.run(tool.execute(file_path=file_path))
            
            if result.status.value == 'success':
                # Get the content
                content = result.output if hasattr(result, 'output') else result.data
                
                # If it's requirements.txt, provide a summary
                if 'requirements' in file_path.lower():
                    # Remove line numbers from tool output
                    lines = []
                    for line in content.split('\n'):
                        # Remove line number prefix (e.g., "   1→")
                        if '→' in line:
                            lines.append(line.split('→', 1)[1])
                        else:
                            lines.append(line)
                    
                    non_empty_lines = [l.strip() for l in lines if l.strip() and not l.strip().startswith('#')]
                    
                    print(f"\n📋 Summary of {file_path}:")
                    print(f"   Total dependencies: {len(non_empty_lines)}")
                    print("\n   Categories found:")
                    
                    current_category = "Uncategorized"
                    categories = {}
                    
                    for line in lines:
                        if line.strip().startswith('# ==='):
                            current_category = line.strip('# =').strip()
                        elif line.strip() and not line.strip().startswith('#'):
                            if current_category not in categories:
                                categories[current_category] = []
                            categories[current_category].append(line.strip())
                    
                    for cat, deps in categories.items():
                        if deps:
                            print(f"   • {cat}: {len(deps)} packages")
                            for dep in deps[:3]:  # Show first 3
                                pkg = dep.split('>=')[0].split('==')[0].split('[')[0]
                                print(f"     - {pkg}")
                            if len(deps) > 3:
                                print(f"     ... and {len(deps)-3} more")
                    
                    print("\n✅ File successfully read and summarized!")
                else:
                    # Show first part of file
                    lines = content.split('\n')
                    print(f"\n📄 Content of {file_path} ({len(lines)} lines):")
                    print("-" * 50)
                    for line in lines[:30]:  # Show first 30 lines
                        print(line)
                    if len(lines) > 30:
                        print(f"... ({len(lines) - 30} more lines)")
                    print("-" * 50)
            else:
                print(f"❌ Error: {result.output if hasattr(result, 'output') else 'Could not read file'}")
            
            return True
    
    # Check for list files request
    elif any(phrase in message_lower for phrase in ['list files', 'ls', 'show directory', 'what files']):
        dir_match = re.search(r'(?:in|at)\s+([~/\.\w\-_/]+)', message)
        directory = dir_match.group(1) if dir_match else '.'
        
        print(f"\n🔧 Listing directory: {directory}")
        
        tool = ListFilesTool()
        result = asyncio.run(tool.execute(directory=directory))
        
        if result.status.value == 'success':
            print(result.output if hasattr(result, 'output') else result.data)
        else:
            print(f"❌ Error: {result.output if hasattr(result, 'output') else 'Could not list directory'}")
        
        return True
    
    # Check for bash command request
    elif any(phrase in message_lower for phrase in ['run command', 'execute', 'bash']):
        cmd_match = re.search(r'(?:run|execute|bash)\s+(.+)', message, re.IGNORECASE)
        if cmd_match:
            command = cmd_match.group(1).strip('"\'')
            
            print(f"\n🔧 Running command: {command}")
            
            tool = RunCommandTool()
            result = asyncio.run(tool.execute(command=command))
            
            if result.status.value == 'success':
                print(result.output if hasattr(result, 'output') else result.data)
            else:
                print(f"❌ Error: {result.output if hasattr(result, 'output') else 'Command failed'}")
            
            return True
    
    return False


def main():
    """Simple ADAM CLI with tools"""
    
    print("\n" + "="*60)
    print("ADAM - Advanced Data Analytics Model")
    print("Now with File Access and Tool Support!")
    print("="*60)
    print("\nExample commands:")
    print("  • 'read file /path/to/file.txt'")
    print("  • 'list files in .'")
    print("  • 'run command ls -la'")
    print("  • 'exit' to quit\n")
    
    while True:
        try:
            user_input = input("You: ").strip()
            
            if user_input.lower() in ['exit', 'quit']:
                print("\nGoodbye!")
                break
            
            if not user_input:
                continue
            
            # Try to process with tools
            if not process_message_with_tools(user_input):
                print("\nℹ️  No tool detected. In full ADAM, this would go to the AI for response.")
                print("   Try commands like: 'read file requirements.txt' or 'list files'")
        
        except KeyboardInterrupt:
            print("\n\nInterrupted. Type 'exit' to quit.")
        except Exception as e:
            print(f"\n❌ Error: {e}")


if __name__ == "__main__":
    main()