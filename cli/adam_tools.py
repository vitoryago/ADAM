#!/usr/bin/env python3
"""
ADAM CLI with Tool Support - File system access like Claude Code
"""

import asyncio
import sys
from pathlib import Path
import os
from typing import Optional, Dict, Any

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.adam.memory import ADAMMemoryAdvanced
from src.adam.conversation_system import ConversationSystem
from src.adam.llm.client import UnifiedLLMClient
from src.adam.llm.config import LLMConfig

# Import tools
from src.adam.tools.file_tools import (
    ReadFileTool, WriteFileTool, EditFileTool,
    ListFilesTool, SearchFilesTool
)
from src.adam.tools.system_tools import RunCommandTool

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
    print("Install 'rich' for better formatting: pip install rich")


class ADAMToolChat:
    """ADAM CLI with full tool support - like Claude Code"""
    
    def __init__(self):
        print("Initializing ADAM with Tools...")
        
        # Initialize memory system
        self.memory = ADAMMemoryAdvanced()
        
        # Initialize conversation system
        self.conversation = ConversationSystem()
        self.session_id = self.conversation.start_session("Tool-Enabled Session")
        
        # Initialize LLM client
        self.llm_client = UnifiedLLMClient()
        
        # Initialize tools
        self.tools = {
            'read': ReadFileTool(),
            'write': WriteFileTool(),
            'edit': EditFileTool(),
            'list': ListFilesTool(),
            'search': SearchFilesTool(),
            'bash': RunCommandTool(),
        }
        
        print("✅ ADAM is ready with full tool support!")
        print("📁 File access enabled")
        print("🔧 Bash commands enabled")
        print("🧠 Memory system active\n")
    
    def detect_tool_request(self, message: str) -> Optional[Dict[str, Any]]:
        """Detect if message requires tool usage"""
        
        message_lower = message.lower()
        
        # File reading patterns
        if any(phrase in message_lower for phrase in [
            'read file', 'read the file', 'show file', 'show me the file',
            'open file', 'cat ', 'view file', 'check the file',
            'look at', 'examine'
        ]):
            # Extract file path
            import re
            # Look for paths like /Users/... or ./... or ~/...
            path_match = re.search(r'[~/\.]?[/\w\-_\.]+\.\w+', message)
            if path_match:
                file_path = path_match.group()
                if file_path.startswith('~'):
                    file_path = os.path.expanduser(file_path)
                return {
                    'tool': 'read',
                    'params': {'file_path': file_path}
                }
        
        # Directory listing patterns
        if any(phrase in message_lower for phrase in [
            'list files', 'ls ', 'dir ', 'show directory',
            'what files', 'show folder'
        ]):
            # Extract directory path or use current
            import re
            dir_match = re.search(r'(?:in|of|at)\s+([~/\.\w\-_/]+)', message)
            directory = dir_match.group(1) if dir_match else '.'
            return {
                'tool': 'list',
                'params': {'directory': directory}
            }
        
        # Bash command patterns
        if any(phrase in message_lower for phrase in [
            'run command', 'execute', 'run bash', 'shell command'
        ]):
            # Extract command
            import re
            cmd_match = re.search(r'(?:run|execute|bash)\s+["\']?(.+?)["\']?$', message, re.IGNORECASE)
            if cmd_match:
                return {
                    'tool': 'bash',
                    'params': {'command': cmd_match.group(1)}
                }
        
        return None
    
    def execute_tool(self, tool_info: Dict[str, Any]) -> str:
        """Execute a tool and return formatted result"""
        
        tool_name = tool_info['tool']
        params = tool_info['params']
        
        if tool_name not in self.tools:
            return f"❌ Unknown tool: {tool_name}"
        
        tool = self.tools[tool_name]
        result = tool.execute(**params)
        
        # Format the result nicely
        if hasattr(result, 'status'):
            if result.status.value == 'success':
                if tool_name == 'read' and RICH_AVAILABLE:
                    # Show file content with syntax highlighting
                    file_path = params.get('file_path', '')
                    if file_path.endswith(('.py', '.js', '.ts', '.jsx', '.tsx', '.json', '.yaml', '.yml', '.md')):
                        # Get file extension for syntax highlighting
                        ext = Path(file_path).suffix[1:]
                        if ext in ['yml']: ext = 'yaml'
                        if ext in ['ts', 'tsx', 'jsx']: ext = 'javascript'
                        
                        # Get raw content without line numbers
                        with open(file_path, 'r') as f:
                            raw_content = f.read()
                        
                        syntax = Syntax(raw_content, ext, theme="monokai", line_numbers=True)
                        console.print(Panel(syntax, title=f"📄 {file_path}", border_style="blue"))
                        return ""  # Already printed
                    else:
                        return f"📄 Content of {file_path}:\n{result.output if hasattr(result, 'output') else result.data}"
                else:
                    return f"✅ {result.output if hasattr(result, 'output') else result.data}"
            else:
                return f"❌ {result.message if hasattr(result, 'message') else 'Error occurred'}"
        
        return str(result)
    
    async def process_message(self, message: str) -> str:
        """Process a message, execute tools if needed, then get AI response"""
        
        # Check for tool request
        tool_info = self.detect_tool_request(message)
        
        tool_output = ""
        if tool_info:
            print(f"🔧 Executing tool: {tool_info['tool']}...")
            tool_output = self.execute_tool(tool_info)
            if tool_output:
                print(tool_output)
        
        # Now get AI response with context about what was done
        context = ""
        if tool_output:
            context = f"\n\nTool execution result:\n{tool_output}\n"
        
        # Search memories
        memories = self.memory.search_memories(message, limit=3)
        memory_context = ""
        if memories:
            memory_context = "\n\nRelevant memories:\n"
            for mem in memories[:2]:
                memory_context += f"- {mem.content[:100]}...\n"
        
        # Get AI response
        full_message = message + context
        response = await self.llm_client.generate(
            message=full_message,
            context=memory_context,
            model="automatic"  # Let it choose based on complexity
        )
        
        # Store in memory if significant
        if len(response.content.split()) > 50 or tool_output:
            self.memory.store(
                content=f"User: {message}\nTool: {tool_info['tool'] if tool_info else 'none'}\nAI: {response.content}",
                metadata={
                    'session_id': self.session_id,
                    'has_tool': bool(tool_info),
                    'tool': tool_info['tool'] if tool_info else None
                }
            )
        
        return response.content
    
    async def run(self):
        """Main chat loop"""
        
        print("\n" + "="*60)
        print("ADAM - Advanced Data Analytics Model with Tools")
        print("Your AI assistant with file access and perfect memory")
        print("="*60)
        print("\nCommands:")
        print("  'exit' or 'quit' - End conversation")
        print("  'help' - Show available tools\n")
        
        while True:
            try:
                user_input = input("\nYou: ").strip()
                
                if user_input.lower() in ['exit', 'quit']:
                    print("\nGoodbye! Your conversation has been saved.")
                    self.conversation.end_session(self.session_id)
                    break
                
                if user_input.lower() == 'help':
                    print("\n📚 Available Tools:")
                    print("  • Read files: 'read [file_path]' or 'show me [file]'")
                    print("  • List directory: 'list files in [directory]' or 'ls'")
                    print("  • Search files: 'search for [pattern] in [directory]'")
                    print("  • Run commands: 'run command [command]' or 'execute [command]'")
                    print("  • Write files: 'write [content] to [file]'")
                    print("  • Edit files: 'edit [file] replace [old] with [new]'")
                    continue
                
                if not user_input:
                    continue
                
                # Process message (with tools if needed)
                response = await self.process_message(user_input)
                
                if RICH_AVAILABLE and response:
                    console.print(Panel(Markdown(response), title="ADAM", border_style="green"))
                else:
                    print(f"\nADAM: {response}")
                
            except KeyboardInterrupt:
                print("\n\nInterrupted. Type 'exit' to quit properly.")
            except Exception as e:
                print(f"\n❌ Error: {e}")


def main():
    """Entry point"""
    chat = ADAMToolChat()
    asyncio.run(chat.run())


if __name__ == "__main__":
    main()