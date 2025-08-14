"""
Code Generation Tools for ADAM
Provides code generation, DAG creation, and optimization capabilities
"""

from pathlib import Path
from typing import Dict, Any, List, Optional
from .base import Tool, ToolResult, ToolStatus
from ..llm.client import UnifiedLLMClient
from ..llm.config import LLMConfig
import json
import ast

class GenerateCodeTool(Tool):
    """Generate code based on requirements"""
    
    def __init__(self, llm_client: Optional[UnifiedLLMClient] = None):
        super().__init__(
            name="generate_code",
            description="Generate code based on specifications"
        )
        self.llm_client = llm_client or UnifiedLLMClient(LLMConfig())
    
    async def execute(self,
                     requirements: str,
                     language: str = "python",
                     template_file: Optional[str] = None,
                     output_file: Optional[str] = None) -> ToolResult:
        """Generate code based on requirements"""
        try:
            # Read template if provided
            template_code = ""
            if template_file:
                template_path = Path(template_file)
                if template_path.exists():
                    with open(template_path, 'r') as f:
                        template_code = f.read()
                    requirements = f"Create code similar to this template:\n```{language}\n{template_code}\n```\n\nRequirements: {requirements}"
            
            # Create prompt
            prompt = f"""Generate {language} code for the following requirements:
{requirements}

Provide clean, well-commented, production-ready code.
Only return the code, no explanations."""
            
            # Generate code using LLM
            response = await self.llm_client.complete(
                prompt=prompt,
                model="grok-4",  # Use powerful model for code generation
                temperature=0.3,  # Lower temperature for more consistent code
                max_tokens=4000
            )
            
            generated_code = response.content
            
            # Extract code block if wrapped in markdown
            if "```" in generated_code:
                import re
                code_blocks = re.findall(r'```(?:\w+)?\n(.*?)```', generated_code, re.DOTALL)
                if code_blocks:
                    generated_code = code_blocks[0]
            
            # Save to file if specified
            if output_file:
                output_path = Path(output_file)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                with open(output_path, 'w') as f:
                    f.write(generated_code)
                
                return ToolResult(
                    status=ToolStatus.SUCCESS,
                    data=generated_code,
                    message=f"Generated {language} code and saved to {output_file}",
                    metadata={
                        "language": language,
                        "output_file": str(output_path),
                        "lines": len(generated_code.split('\n'))
                    }
                )
            
            return ToolResult(
                status=ToolStatus.SUCCESS,
                data=generated_code,
                message=f"Generated {language} code",
                metadata={
                    "language": language,
                    "lines": len(generated_code.split('\n'))
                }
            )
        except Exception as e:
            return ToolResult(
                status=ToolStatus.ERROR,
                data=None,
                message=f"Code generation failed: {str(e)}"
            )
    
    def validate_params(self, **kwargs) -> bool:
        return 'requirements' in kwargs
    
    def _get_param_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "requirements": {"type": "string", "description": "Code requirements or description"},
                "language": {"type": "string", "description": "Programming language"},
                "template_file": {"type": "string", "description": "Template file to base code on"},
                "output_file": {"type": "string", "description": "File to save generated code"}
            },
            "required": ["requirements"]
        }

class CreateDAGTool(Tool):
    """Create Apache Airflow DAGs"""
    
    def __init__(self, llm_client: Optional[UnifiedLLMClient] = None):
        super().__init__(
            name="create_dag",
            description="Create Apache Airflow DAG files"
        )
        self.llm_client = llm_client or UnifiedLLMClient(LLMConfig())
    
    async def execute(self,
                     dag_name: str,
                     tasks: List[Dict[str, Any]],
                     schedule: str = "@daily",
                     output_file: Optional[str] = None) -> ToolResult:
        """Create an Airflow DAG"""
        try:
            # Build task descriptions
            task_descriptions = []
            for task in tasks:
                desc = f"- Task: {task.get('name', 'unnamed')}"
                if 'type' in task:
                    desc += f" (Type: {task['type']})"
                if 'dependencies' in task:
                    desc += f" (Depends on: {', '.join(task['dependencies'])})"
                if 'description' in task:
                    desc += f"\n  Description: {task['description']}"
                task_descriptions.append(desc)
            
            # Create prompt
            prompt = f"""Create an Apache Airflow DAG with the following specifications:

DAG Name: {dag_name}
Schedule: {schedule}

Tasks:
{chr(10).join(task_descriptions)}

Generate a complete, production-ready Airflow DAG file with:
- Proper imports
- DAG configuration with retries and error handling
- All specified tasks with appropriate operators
- Task dependencies
- Documentation strings
- Best practices for Airflow 2.x

Return only the Python code."""
            
            # Generate DAG code
            response = await self.llm_client.complete(
                prompt=prompt,
                model="grok-4",
                temperature=0.3,
                max_tokens=4000
            )
            
            dag_code = response.content
            
            # Extract code block if wrapped
            if "```" in dag_code:
                import re
                code_blocks = re.findall(r'```(?:python)?\n(.*?)```', dag_code, re.DOTALL)
                if code_blocks:
                    dag_code = code_blocks[0]
            
            # Save to file if specified
            if output_file:
                output_path = Path(output_file)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                with open(output_path, 'w') as f:
                    f.write(dag_code)
                
                return ToolResult(
                    status=ToolStatus.SUCCESS,
                    data=dag_code,
                    message=f"Created DAG '{dag_name}' and saved to {output_file}",
                    metadata={
                        "dag_name": dag_name,
                        "task_count": len(tasks),
                        "schedule": schedule,
                        "output_file": str(output_path)
                    }
                )
            
            return ToolResult(
                status=ToolStatus.SUCCESS,
                data=dag_code,
                message=f"Created DAG '{dag_name}'",
                metadata={
                    "dag_name": dag_name,
                    "task_count": len(tasks),
                    "schedule": schedule
                }
            )
        except Exception as e:
            return ToolResult(
                status=ToolStatus.ERROR,
                data=None,
                message=f"DAG creation failed: {str(e)}"
            )
    
    def validate_params(self, **kwargs) -> bool:
        return 'dag_name' in kwargs and 'tasks' in kwargs
    
    def _get_param_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "dag_name": {"type": "string", "description": "Name of the DAG"},
                "tasks": {"type": "array", "description": "List of task definitions"},
                "schedule": {"type": "string", "description": "Schedule interval (cron or preset)"},
                "output_file": {"type": "string", "description": "File to save DAG code"}
            },
            "required": ["dag_name", "tasks"]
        }

class OptimizeSQLTool(Tool):
    """Optimize SQL queries"""
    
    def __init__(self, llm_client: Optional[UnifiedLLMClient] = None):
        super().__init__(
            name="optimize_sql",
            description="Optimize SQL queries for performance"
        )
        self.llm_client = llm_client or UnifiedLLMClient(LLMConfig())
    
    async def execute(self,
                     query: str,
                     dialect: str = "postgresql",
                     context: Optional[str] = None) -> ToolResult:
        """Optimize SQL query"""
        try:
            # Create optimization prompt
            prompt = f"""Optimize this {dialect} SQL query for performance:

```sql
{query}
```

{f"Context: {context}" if context else ""}

Provide:
1. The optimized query
2. Explanation of optimizations made
3. Performance improvement estimates

Return the optimized query and explanations in a structured format."""
            
            # Get optimization from LLM
            response = await self.llm_client.complete(
                prompt=prompt,
                model="grok-4",
                temperature=0.2,
                max_tokens=2000
            )
            
            # Parse response
            optimized_content = response.content
            
            # Extract optimized query
            import re
            sql_blocks = re.findall(r'```sql\n(.*?)```', optimized_content, re.DOTALL)
            optimized_query = sql_blocks[0] if sql_blocks else query
            
            # Extract explanations
            explanations = []
            lines = optimized_content.split('\n')
            for i, line in enumerate(lines):
                if line.strip().startswith(('1.', '2.', '3.', '-', '*')):
                    explanations.append(line.strip())
            
            return ToolResult(
                status=ToolStatus.SUCCESS,
                data={
                    "original": query,
                    "optimized": optimized_query,
                    "explanations": explanations,
                    "full_analysis": optimized_content
                },
                message="SQL query optimized successfully",
                metadata={
                    "dialect": dialect,
                    "original_length": len(query),
                    "optimized_length": len(optimized_query)
                }
            )
        except Exception as e:
            return ToolResult(
                status=ToolStatus.ERROR,
                data=None,
                message=f"SQL optimization failed: {str(e)}"
            )
    
    def validate_params(self, **kwargs) -> bool:
        return 'query' in kwargs
    
    def _get_param_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "SQL query to optimize"},
                "dialect": {"type": "string", "description": "SQL dialect (postgresql, mysql, etc.)"},
                "context": {"type": "string", "description": "Additional context about the database"}
            },
            "required": ["query"]
        }

class CreateProjectStructureTool(Tool):
    """Create complete project structure"""
    
    def __init__(self):
        super().__init__(
            name="create_project",
            description="Create a complete project structure with files"
        )
    
    async def execute(self,
                     project_name: str,
                     project_type: str,
                     base_path: str = ".",
                     features: Optional[List[str]] = None) -> ToolResult:
        """Create project structure"""
        try:
            base = Path(base_path) / project_name
            base.mkdir(parents=True, exist_ok=True)
            
            created_files = []
            
            if project_type == "python":
                # Python project structure
                structure = {
                    "src": {
                        project_name: {
                            "__init__.py": "# Main package",
                            "main.py": self._get_python_main(),
                            "config.py": "# Configuration",
                            "utils.py": "# Utilities"
                        }
                    },
                    "tests": {
                        "__init__.py": "",
                        "test_main.py": self._get_python_test()
                    },
                    "requirements.txt": "# Project dependencies",
                    "setup.py": self._get_python_setup(project_name),
                    "README.md": f"# {project_name}\n\n## Description\n\n## Installation\n\n## Usage",
                    ".gitignore": self._get_gitignore("python")
                }
            elif project_type == "typescript":
                # TypeScript project structure
                structure = {
                    "src": {
                        "index.ts": self._get_typescript_main(),
                        "types.ts": "// Type definitions",
                        "utils.ts": "// Utilities"
                    },
                    "tests": {
                        "index.test.ts": "// Tests"
                    },
                    "package.json": self._get_package_json(project_name),
                    "tsconfig.json": self._get_tsconfig(),
                    "README.md": f"# {project_name}\n\n## Description\n\n## Installation\n\n## Usage",
                    ".gitignore": self._get_gitignore("node")
                }
            elif project_type == "react":
                # React project structure
                structure = {
                    "src": {
                        "App.tsx": self._get_react_app(),
                        "index.tsx": self._get_react_index(),
                        "components": {
                            ".gitkeep": ""
                        },
                        "styles": {
                            "App.css": "/* App styles */"
                        }
                    },
                    "public": {
                        "index.html": self._get_react_html()
                    },
                    "package.json": self._get_react_package_json(project_name),
                    "tsconfig.json": self._get_tsconfig(),
                    "README.md": f"# {project_name}\n\n## React Application",
                    ".gitignore": self._get_gitignore("node")
                }
            else:
                # Generic structure
                structure = {
                    "src": {},
                    "docs": {},
                    "tests": {},
                    "README.md": f"# {project_name}"
                }
            
            # Create files recursively
            def create_structure(parent_path: Path, struct: Dict):
                for name, content in struct.items():
                    path = parent_path / name
                    if isinstance(content, dict):
                        path.mkdir(exist_ok=True)
                        create_structure(path, content)
                    else:
                        path.write_text(content)
                        created_files.append(str(path.relative_to(base)))
            
            create_structure(base, structure)
            
            return ToolResult(
                status=ToolStatus.SUCCESS,
                data='\n'.join(created_files),
                message=f"Created {project_type} project '{project_name}' with {len(created_files)} files",
                metadata={
                    "project_path": str(base),
                    "project_type": project_type,
                    "files_created": len(created_files)
                }
            )
        except Exception as e:
            return ToolResult(
                status=ToolStatus.ERROR,
                data=None,
                message=f"Project creation failed: {str(e)}"
            )
    
    def _get_python_main(self) -> str:
        return '''"""Main module"""

def main():
    """Main function"""
    print("Hello, World!")

if __name__ == "__main__":
    main()
'''
    
    def _get_python_test(self) -> str:
        return '''"""Test module"""
import unittest

class TestMain(unittest.TestCase):
    def test_example(self):
        self.assertTrue(True)

if __name__ == "__main__":
    unittest.main()
'''
    
    def _get_python_setup(self, name: str) -> str:
        return f'''from setuptools import setup, find_packages

setup(
    name="{name}",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={{"": "src"}},
    install_requires=[],
)
'''
    
    def _get_typescript_main(self) -> str:
        return '''export function main(): void {
    console.log("Hello, TypeScript!");
}

main();
'''
    
    def _get_package_json(self, name: str) -> str:
        return json.dumps({
            "name": name,
            "version": "1.0.0",
            "main": "dist/index.js",
            "scripts": {
                "build": "tsc",
                "start": "node dist/index.js",
                "dev": "ts-node src/index.ts"
            },
            "devDependencies": {
                "typescript": "^5.0.0",
                "@types/node": "^20.0.0",
                "ts-node": "^10.0.0"
            }
        }, indent=2)
    
    def _get_react_package_json(self, name: str) -> str:
        return json.dumps({
            "name": name,
            "version": "0.1.0",
            "private": True,
            "dependencies": {
                "react": "^18.2.0",
                "react-dom": "^18.2.0"
            },
            "scripts": {
                "start": "vite",
                "build": "vite build",
                "preview": "vite preview"
            },
            "devDependencies": {
                "@types/react": "^18.2.0",
                "@types/react-dom": "^18.2.0",
                "@vitejs/plugin-react": "^4.0.0",
                "typescript": "^5.0.0",
                "vite": "^4.0.0"
            }
        }, indent=2)
    
    def _get_tsconfig(self) -> str:
        return json.dumps({
            "compilerOptions": {
                "target": "ES2020",
                "module": "commonjs",
                "lib": ["ES2020"],
                "outDir": "./dist",
                "rootDir": "./src",
                "strict": True,
                "esModuleInterop": True,
                "skipLibCheck": True,
                "forceConsistentCasingInFileNames": True
            }
        }, indent=2)
    
    def _get_react_app(self) -> str:
        return '''import React from 'react';
import './styles/App.css';

function App() {
    return (
        <div className="App">
            <h1>Welcome to React</h1>
        </div>
    );
}

export default App;
'''
    
    def _get_react_index(self) -> str:
        return '''import React from 'react';
import ReactDOM from 'react-dom/client';
import App from './App';

const root = ReactDOM.createRoot(
    document.getElementById('root') as HTMLElement
);
root.render(
    <React.StrictMode>
        <App />
    </React.StrictMode>
);
'''
    
    def _get_react_html(self) -> str:
        return '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>React App</title>
</head>
<body>
    <div id="root"></div>
</body>
</html>
'''
    
    def _get_gitignore(self, type: str) -> str:
        if type == "python":
            return '''__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
.venv
.env
*.egg-info/
dist/
build/
.pytest_cache/
.coverage
'''
        elif type == "node":
            return '''node_modules/
dist/
build/
.env
.env.local
npm-debug.log*
yarn-debug.log*
yarn-error.log*
.DS_Store
'''
        return ""
    
    def validate_params(self, **kwargs) -> bool:
        return 'project_name' in kwargs and 'project_type' in kwargs
    
    def _get_param_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "project_name": {"type": "string", "description": "Name of the project"},
                "project_type": {"type": "string", "description": "Type of project (python, typescript, react)"},
                "base_path": {"type": "string", "description": "Base path for project creation"},
                "features": {"type": "array", "description": "Additional features to include"}
            },
            "required": ["project_name", "project_type"]
        }