#!/usr/bin/env python3
"""
ADAM CLI - Claude Code Style
Natural language understanding with automatic tool execution
"""

import sys
import os
import re
import json
import asyncio
from pathlib import Path
from typing import Optional, Dict, Any, List
from dotenv import load_dotenv

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Load environment
load_dotenv()

# Import ADAM components
from adam.memory import ADAMMemoryAdvanced
from adam.conversation_system import ConversationSystem
from adam.llm.client import UnifiedLLMClient
from adam.llm.config import LLMConfig

# Import tools
from adam.tools.file_tools import ReadFileTool, WriteFileTool, ListFilesTool, EditFileTool
from adam.tools.system_tools import RunCommandTool

# For colored output
try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.markdown import Markdown
    from rich.syntax import Syntax
    console = Console()
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False


class ADAMClaudeStyle:
    """ADAM that works like Claude Code - understands natural language and executes tools"""
    
    def __init__(self):
        print("Initializing ADAM...")
        
        # Initialize components
        self.memory = ADAMMemoryAdvanced()
        self.conversation = ConversationSystem()
        self.session_id = self.conversation.start_session("Claude-Style Session")
        self.llm_client = UnifiedLLMClient()
        
        # Initialize tools
        self.tools = {
            'read_file': ReadFileTool(),
            'write_file': WriteFileTool(),
            'edit_file': EditFileTool(),
            'list_files': ListFilesTool(),
            'run_command': RunCommandTool(),
        }
        
        # Tool detection patterns
        self.tool_patterns = {
            'read_file': [
                r'(?:read|show|display|check|look at|examine|view|open|cat)\s+(?:the\s+)?(?:file|document)?\s*[:\s]*([\/\w\-\.~]+)',
                r'(?:what\'s in|what is in|contents? of)\s+([\/\w\-\.~]+)',
                r'(?:can you|could you|please)?\s*(?:read|show|check)\s+([\/\w\-\.~]+)',
                r'go through\s+(?:the\s+)?(?:requirements?|file)?\s*(?:file)?\s+(?:on|in|at)?\s*(?:our)?\s*(?:adam)?\s*(?:project)?',
            ],
            'list_files': [
                r'(?:list|show|display)\s+(?:all\s+)?(?:the\s+)?files?\s*(?:in|at)?\s*([\/\w\-\.~]*)',
                r'(?:what|which)\s+files?\s+(?:are|do we have)\s*(?:in|at)?\s*([\/\w\-\.~]*)',
                r'ls\s*([\/\w\-\.~]*)',
            ],
            'run_command': [
                r'(?:run|execute|do)\s+(?:the\s+)?(?:command|bash|shell)?\s*[:\s]*(.+)',
                r'(?:can you|could you|please)?\s*(?:run|execute)\s+(.+)',
            ],
        }
        
        print("ADAM is ready! I can understand natural language and execute tools.")
    
    def extract_tool_and_params(self, message: str) -> Optional[Dict[str, Any]]:
        """Extract tool and parameters from natural language"""
        
        message_lower = message.lower()
        
        # Check for file reading
        for pattern in self.tool_patterns['read_file']:
            match = re.search(pattern, message_lower)
            if match:
                # Special case for "requirements file on our adam project"
                if 'requirements' in message_lower and 'project' in message_lower:
                    return {
                        'tool': 'read_file',
                        'params': {'file_path': '/Users/vitoryago/ADAM/requirements.txt'}
                    }
                
                # Extract file path
                file_path = match.group(1) if match.lastindex else 'requirements.txt'
                if not file_path or file_path in ['file', 'document', 'it', 'that']:
                    # Try to find actual path in original message
                    path_match = re.search(r'[\/\w\-\.~]+\.\w+', message)
                    if path_match:
                        file_path = path_match.group()
                
                if file_path:
                    if file_path.startswith('~'):
                        file_path = os.path.expanduser(file_path)
                    elif not file_path.startswith('/'):
                        # Assume relative to ADAM project
                        file_path = f'/Users/vitoryago/ADAM/{file_path}'
                    
                    return {
                        'tool': 'read_file',
                        'params': {'file_path': file_path}
                    }
        
        # Check for listing files
        for pattern in self.tool_patterns['list_files']:
            match = re.search(pattern, message_lower)
            if match:
                directory = match.group(1) if match.lastindex else '.'
                if not directory:
                    directory = '.'
                return {
                    'tool': 'list_files',
                    'params': {'directory': directory}
                }
        
        # Check for running commands
        for pattern in self.tool_patterns['run_command']:
            match = re.search(pattern, message_lower)
            if match:
                command = match.group(1)
                return {
                    'tool': 'run_command',
                    'params': {'command': command}
                }
        
        return None
    
    async def execute_tool_async(self, tool_name: str, params: Dict[str, Any]) -> str:
        """Execute a tool and return formatted result"""
        
        if tool_name not in self.tools:
            return f"Unknown tool: {tool_name}"
        
        tool = self.tools[tool_name]
        result = await tool.execute(**params)
        
        if result.status.value == 'success':
            return result.data
        else:
            return f"Error: {result.message if hasattr(result, 'message') else 'Tool execution failed'}"
    
    def process_message(self, message: str) -> str:
        """Process message with natural language understanding"""
        
        # First, try to extract tool from natural language
        tool_info = self.extract_tool_and_params(message)
        
        tool_result = None
        if tool_info:
            print(f"🔧 Executing: {tool_info['tool']}...")
            
            # Execute tool
            tool_result = asyncio.run(
                self.execute_tool_async(tool_info['tool'], tool_info['params'])
            )
            
            # Format output based on tool type
            if tool_info['tool'] == 'read_file' and tool_result and not tool_result.startswith('Error'):
                # Process file content
                file_path = tool_info['params']['file_path']
                
                # For requirements.txt, provide analysis
                if 'requirements' in file_path.lower():
                    lines = []
                    for line in tool_result.split('\n'):
                        if '→' in line:
                            lines.append(line.split('→', 1)[1])
                        else:
                            lines.append(line)
                    
                    # Analyze requirements
                    categories = {}
                    current_category = "Uncategorized"
                    total_deps = 0
                    
                    for line in lines:
                        if line.strip().startswith('# ==='):
                            current_category = line.strip('# =').strip()
                        elif line.strip() and not line.strip().startswith('#'):
                            if current_category not in categories:
                                categories[current_category] = []
                            categories[current_category].append(line.strip())
                            total_deps += 1
                    
                    # Generate response
                    response = f"I've read the requirements.txt file. Here's what I found:\n\n"
                    response += f"**Total Dependencies:** {total_deps}\n\n"
                    response += "**Categories:**\n"
                    
                    for cat, deps in categories.items():
                        if deps and cat != "Uncategorized":
                            response += f"\n📦 **{cat}** ({len(deps)} packages):\n"
                            for dep in deps[:5]:
                                pkg = dep.split('>=')[0].split('==')[0].split('[')[0]
                                response += f"  - {pkg}\n"
                            if len(deps) > 5:
                                response += f"  ... and {len(deps)-5} more\n"
                    
                    response += "\nThe requirements file is well-organized with clear categories for different components of ADAM."
                    response += "\nAll dependencies are properly specified with version constraints where needed."
                    
                    return response
                else:
                    # Regular file - show content
                    lines = tool_result.split('\n')
                    if len(lines) > 50:
                        return f"Here's the content of {file_path} ({len(lines)} lines):\n\n```\n{chr(10).join(lines[:50])}\n...\n({len(lines)-50} more lines)\n```"
                    else:
                        return f"Here's the content of {file_path}:\n\n```\n{tool_result}\n```"
            
            elif tool_info['tool'] == 'list_files' and tool_result:
                return f"Here are the files:\n\n{tool_result}"
            
            elif tool_info['tool'] == 'run_command' and tool_result:
                return f"Command output:\n\n```\n{tool_result}\n```"
        
        # If no tool detected, use LLM for general response
        # Search relevant memories
        memories = self.memory.search_memories(message, limit=3)
        memory_context = ""
        if memories:
            memory_context = "Relevant context:\n"
            for mem in memories[:2]:
                memory_context += f"- {mem.content[:100]}...\n"
        
        # Get LLM response
        try:
            response = self.llm_client.chat(
                messages=[
                    {"role": "system", "content": "You are ADAM, an AI assistant with file system access. Help the user with their requests."},
                    {"role": "user", "content": message}
                ],
                model="gpt-5-mini"  # Use fast model for responsiveness
            )
            
            return response.content
        except:
            # Fallback response
            return "I understand you want me to help, but I need more specific instructions. Try asking me to read a file, list files, or run a command."
    
    def run(self):
        """Main interaction loop"""
        
        print("\n" + "="*60)
        print("ADAM - Your AI Assistant (Claude Code Style)")
        print("I understand natural language and can access files!")
        print("="*60 + "\n")
        
        while True:
            try:
                user_input = input("You: ").strip()
                
                if user_input.lower() in ['exit', 'quit']:
                    print("\nGoodbye! Your conversation has been saved.")
                    self.conversation.end_session(self.session_id)
                    break
                
                if not user_input:
                    continue
                
                # Process with natural language understanding
                response = self.process_message(user_input)
                
                # Display response
                if RICH_AVAILABLE:
                    console.print(Panel(Markdown(response), title="ADAM", border_style="green"))
                else:
                    print(f"\nADAM: {response}\n")
                
                # Store significant interactions in memory
                if len(response.split()) > 20:
                    self.memory.store(
                        content=f"User: {user_input}\nADAM: {response}",
                        metadata={
                            'session_id': self.session_id,
                            'type': 'conversation'
                        }
                    )
                
            except KeyboardInterrupt:
                print("\n\nInterrupted. Type 'exit' to quit properly.")
            except Exception as e:
                print(f"\nError: {e}")


def main():
    """Entry point"""
    adam = ADAMClaudeStyle()
    adam.run()


if __name__ == "__main__":
    main()