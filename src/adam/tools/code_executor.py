"""
Code Execution Tool for ADAM
Safe sandboxed code execution with Docker containers
"""

import os
import subprocess
import tempfile
import asyncio
import logging
import json
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
import hashlib
import time

logger = logging.getLogger(__name__)

# Check if Docker is available
try:
    subprocess.run(["docker", "--version"], capture_output=True, check=True)
    DOCKER_AVAILABLE = True
except (subprocess.CalledProcessError, FileNotFoundError):
    DOCKER_AVAILABLE = False
    logger.warning("Docker not available. Code execution will be limited.")


@dataclass
class ExecutionResult:
    """Result of code execution"""
    stdout: str
    stderr: str
    exit_code: int
    execution_time: float
    language: str
    truncated: bool = False
    error: Optional[str] = None
    
    def to_dict(self):
        return {
            'stdout': self.stdout,
            'stderr': self.stderr,
            'exit_code': self.exit_code,
            'execution_time': self.execution_time,
            'language': self.language,
            'truncated': self.truncated,
            'error': self.error
        }
    
    @property
    def success(self) -> bool:
        return self.exit_code == 0 and self.error is None


class CodeExecutor:
    """
    Safe code execution in sandboxed environments
    """
    
    # Docker images for different languages
    DOCKER_IMAGES = {
        'python': 'python:3.11-slim',
        'javascript': 'node:18-slim',
        'typescript': 'node:18-slim',
        'java': 'openjdk:17-slim',
        'cpp': 'gcc:latest',
        'go': 'golang:1.21-alpine',
        'rust': 'rust:slim',
        'ruby': 'ruby:3.2-slim',
        'php': 'php:8.2-cli',
        'sql': 'postgres:15-alpine'
    }
    
    # File extensions to language mapping
    EXTENSIONS = {
        '.py': 'python',
        '.js': 'javascript',
        '.ts': 'typescript',
        '.java': 'java',
        '.cpp': 'cpp',
        '.cc': 'cpp',
        '.go': 'go',
        '.rs': 'rust',
        '.rb': 'ruby',
        '.php': 'php',
        '.sql': 'sql'
    }
    
    def __init__(self, use_docker: bool = True, timeout: int = 30):
        """
        Initialize code executor
        
        Args:
            use_docker: Whether to use Docker for sandboxing
            timeout: Maximum execution time in seconds
        """
        self.use_docker = use_docker and DOCKER_AVAILABLE
        self.timeout = timeout
        self.execution_cache = {}
        
        if self.use_docker:
            self._pull_base_images()
    
    def _pull_base_images(self):
        """Pull commonly used Docker images"""
        images_to_pull = ['python:3.11-slim', 'node:18-slim']
        for image in images_to_pull:
            try:
                logger.info(f"Pulling Docker image: {image}")
                subprocess.run(
                    ["docker", "pull", image],
                    capture_output=True,
                    timeout=60
                )
            except Exception as e:
                logger.warning(f"Failed to pull {image}: {e}")
    
    def detect_language(self, code: str, filename: Optional[str] = None) -> str:
        """Detect programming language from code or filename"""
        if filename:
            ext = Path(filename).suffix.lower()
            if ext in self.EXTENSIONS:
                return self.EXTENSIONS[ext]
        
        # Simple heuristics for language detection
        if 'def ' in code or 'import ' in code or 'print(' in code:
            return 'python'
        elif 'function ' in code or 'const ' in code or 'console.log' in code:
            return 'javascript'
        elif 'public class' in code or 'public static void main' in code:
            return 'java'
        elif '#include' in code or 'int main()' in code:
            return 'cpp'
        elif 'func ' in code or 'package main' in code:
            return 'go'
        elif 'fn main()' in code or 'let mut' in code:
            return 'rust'
        elif 'SELECT' in code.upper() or 'CREATE TABLE' in code.upper():
            return 'sql'
        
        return 'python'  # Default
    
    def execute(self, code: str, language: Optional[str] = None,
                stdin: Optional[str] = None) -> ExecutionResult:
        """
        Execute code in a sandboxed environment
        
        Args:
            code: Code to execute
            language: Programming language (auto-detected if None)
            stdin: Input to provide to the program
            
        Returns:
            ExecutionResult object
        """
        if language is None:
            language = self.detect_language(code)
        
        # Check cache
        cache_key = hashlib.md5(f"{code}:{language}:{stdin}".encode()).hexdigest()
        if cache_key in self.execution_cache:
            logger.info("Returning cached execution result")
            return self.execution_cache[cache_key]
        
        start_time = time.time()
        
        if self.use_docker:
            result = self._execute_docker(code, language, stdin)
        else:
            result = self._execute_local(code, language, stdin)
        
        result.execution_time = time.time() - start_time
        
        # Cache result
        self.execution_cache[cache_key] = result
        
        return result
    
    def _execute_docker(self, code: str, language: str, 
                       stdin: Optional[str]) -> ExecutionResult:
        """Execute code in Docker container"""
        if language not in self.DOCKER_IMAGES:
            return ExecutionResult(
                stdout="",
                stderr=f"Unsupported language: {language}",
                exit_code=1,
                execution_time=0,
                language=language,
                error="Language not supported for Docker execution"
            )
        
        image = self.DOCKER_IMAGES[language]
        
        # Create temporary file with code
        with tempfile.NamedTemporaryFile(mode='w', suffix=self._get_file_extension(language),
                                       delete=False) as f:
            f.write(code)
            code_file = f.name
        
        try:
            # Build Docker command
            docker_cmd = [
                "docker", "run",
                "--rm",  # Remove container after execution
                "-i",    # Interactive (for stdin)
                "--network", "none",  # No network access
                "--memory", "512m",   # Memory limit
                "--cpus", "0.5",      # CPU limit
                "-v", f"{code_file}:/code{self._get_file_extension(language)}",
                image
            ]
            
            # Add language-specific execution command
            exec_cmd = self._get_execution_command(language, f"/code{self._get_file_extension(language)}")
            docker_cmd.extend(exec_cmd)
            
            # Execute
            process = subprocess.run(
                docker_cmd,
                input=stdin.encode() if stdin else None,
                capture_output=True,
                timeout=self.timeout
            )
            
            stdout = process.stdout.decode('utf-8', errors='replace')
            stderr = process.stderr.decode('utf-8', errors='replace')
            
            # Truncate if output is too long
            max_output = 10000
            truncated = False
            if len(stdout) > max_output:
                stdout = stdout[:max_output] + "\n... (output truncated)"
                truncated = True
            if len(stderr) > max_output:
                stderr = stderr[:max_output] + "\n... (output truncated)"
                truncated = True
            
            return ExecutionResult(
                stdout=stdout,
                stderr=stderr,
                exit_code=process.returncode,
                execution_time=0,
                language=language,
                truncated=truncated
            )
            
        except subprocess.TimeoutExpired:
            return ExecutionResult(
                stdout="",
                stderr=f"Execution timed out after {self.timeout} seconds",
                exit_code=-1,
                execution_time=self.timeout,
                language=language,
                error="Timeout"
            )
        except Exception as e:
            return ExecutionResult(
                stdout="",
                stderr=str(e),
                exit_code=-1,
                execution_time=0,
                language=language,
                error=str(e)
            )
        finally:
            # Clean up temporary file
            try:
                os.unlink(code_file)
            except:
                pass
    
    def _execute_local(self, code: str, language: str,
                      stdin: Optional[str]) -> ExecutionResult:
        """Execute code locally (less safe, for development)"""
        logger.warning("Executing code locally without sandboxing")
        
        if language not in ['python', 'javascript', 'sql']:
            return ExecutionResult(
                stdout="",
                stderr="Only Python, JavaScript, and SQL supported without Docker",
                exit_code=1,
                execution_time=0,
                language=language,
                error="Language not supported for local execution"
            )
        
        try:
            if language == 'python':
                # Execute Python code
                import sys
                from io import StringIO
                
                old_stdout = sys.stdout
                old_stderr = sys.stderr
                sys.stdout = StringIO()
                sys.stderr = StringIO()
                
                try:
                    exec(code, {'__name__': '__main__'})
                    stdout = sys.stdout.getvalue()
                    stderr = sys.stderr.getvalue()
                    exit_code = 0
                except Exception as e:
                    stdout = sys.stdout.getvalue()
                    stderr = str(e)
                    exit_code = 1
                finally:
                    sys.stdout = old_stdout
                    sys.stderr = old_stderr
                
                return ExecutionResult(
                    stdout=stdout,
                    stderr=stderr,
                    exit_code=exit_code,
                    execution_time=0,
                    language=language
                )
                
            elif language == 'javascript':
                # Use Node.js if available
                with tempfile.NamedTemporaryFile(mode='w', suffix='.js', delete=False) as f:
                    f.write(code)
                    temp_file = f.name
                
                try:
                    process = subprocess.run(
                        ['node', temp_file],
                        input=stdin.encode() if stdin else None,
                        capture_output=True,
                        timeout=self.timeout
                    )
                    
                    return ExecutionResult(
                        stdout=process.stdout.decode('utf-8', errors='replace'),
                        stderr=process.stderr.decode('utf-8', errors='replace'),
                        exit_code=process.returncode,
                        execution_time=0,
                        language=language
                    )
                finally:
                    os.unlink(temp_file)
                    
            elif language == 'sql':
                # For SQL, just return the query (would need actual DB connection)
                return ExecutionResult(
                    stdout=f"SQL Query:\n{code}\n\n(Actual execution requires database connection)",
                    stderr="",
                    exit_code=0,
                    execution_time=0,
                    language=language
                )
                
        except Exception as e:
            return ExecutionResult(
                stdout="",
                stderr=str(e),
                exit_code=-1,
                execution_time=0,
                language=language,
                error=str(e)
            )
    
    def _get_file_extension(self, language: str) -> str:
        """Get file extension for language"""
        extensions = {
            'python': '.py',
            'javascript': '.js',
            'typescript': '.ts',
            'java': '.java',
            'cpp': '.cpp',
            'go': '.go',
            'rust': '.rs',
            'ruby': '.rb',
            'php': '.php',
            'sql': '.sql'
        }
        return extensions.get(language, '.txt')
    
    def _get_execution_command(self, language: str, filepath: str) -> list:
        """Get execution command for language"""
        commands = {
            'python': ['python', filepath],
            'javascript': ['node', filepath],
            'typescript': ['sh', '-c', f'npx ts-node {filepath}'],
            'java': ['sh', '-c', f'javac {filepath} && java Main'],
            'cpp': ['sh', '-c', f'g++ {filepath} -o /tmp/a.out && /tmp/a.out'],
            'go': ['go', 'run', filepath],
            'rust': ['sh', '-c', f'rustc {filepath} -o /tmp/a.out && /tmp/a.out'],
            'ruby': ['ruby', filepath],
            'php': ['php', filepath],
            'sql': ['echo', 'SQL execution not supported in Docker']
        }
        return commands.get(language, ['cat', filepath])
    
    async def execute_async(self, code: str, language: Optional[str] = None,
                           stdin: Optional[str] = None) -> ExecutionResult:
        """Async version of execute"""
        return await asyncio.to_thread(self.execute, code, language, stdin)
    
    def test_setup(self) -> Dict[str, bool]:
        """Test which languages are available for execution"""
        results = {}
        
        test_codes = {
            'python': 'print("Hello from Python")',
            'javascript': 'console.log("Hello from JavaScript")',
            'go': 'package main\nimport "fmt"\nfunc main() { fmt.Println("Hello from Go") }',
            'rust': 'fn main() { println!("Hello from Rust"); }',
            'ruby': 'puts "Hello from Ruby"',
            'php': '<?php echo "Hello from PHP\\n"; ?>'
        }
        
        for lang, code in test_codes.items():
            try:
                result = self.execute(code, language=lang)
                results[lang] = result.success
            except Exception:
                results[lang] = False
        
        return results


# Convenience function
def run_code(code: str, language: Optional[str] = None) -> Dict[str, Any]:
    """
    Quick function to run code
    
    Example:
        result = run_code('print("Hello World")')
        print(result['stdout'])
    """
    executor = CodeExecutor(use_docker=DOCKER_AVAILABLE)
    result = executor.execute(code, language)
    return result.to_dict()


if __name__ == "__main__":
    # Test the executor
    executor = CodeExecutor()
    
    # Test Python
    print("Testing Python:")
    result = executor.execute("""
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

for i in range(10):
    print(f"fib({i}) = {fibonacci(i)}")
""", language='python')
    print(result.stdout)
    
    # Test JavaScript
    if DOCKER_AVAILABLE:
        print("\nTesting JavaScript:")
        result = executor.execute("""
const factorial = (n) => n <= 1 ? 1 : n * factorial(n - 1);
for (let i = 1; i <= 5; i++) {
    console.log(`${i}! = ${factorial(i)}`);
}
""", language='javascript')
        print(result.stdout)
    
    # Show available languages
    print("\nAvailable languages:")
    for lang, available in executor.test_setup().items():
        print(f"  {lang}: {'✓' if available else '✗'}")