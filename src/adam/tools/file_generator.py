"""
File Generation Tool for ADAM
Create various file types with templates and best practices
"""

import os
import json
import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime
import re

logger = logging.getLogger(__name__)


@dataclass
class GeneratedFile:
    """Represents a generated file"""
    filename: str
    content: str
    file_type: str
    path: Optional[Path] = None
    created_at: datetime = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.metadata is None:
            self.metadata = {}
    
    def save(self, directory: Path = Path.cwd()) -> Path:
        """Save file to disk"""
        full_path = directory / self.filename
        full_path.parent.mkdir(parents=True, exist_ok=True)
        full_path.write_text(self.content)
        self.path = full_path
        return full_path
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'filename': self.filename,
            'content': self.content,
            'file_type': self.file_type,
            'path': str(self.path) if self.path else None,
            'created_at': self.created_at.isoformat(),
            'metadata': self.metadata
        }


class FileGenerator:
    """
    Generate various types of files with appropriate templates
    """
    
    def __init__(self):
        self.templates = self._load_templates()
        self.generated_files = []
    
    def _load_templates(self) -> Dict[str, str]:
        """Load file templates"""
        return {
            'python_script': self._python_template(),
            'python_class': self._python_class_template(),
            'python_test': self._python_test_template(),
            'javascript_module': self._javascript_module_template(),
            'react_component': self._react_component_template(),
            'docker': self._dockerfile_template(),
            'docker_compose': self._docker_compose_template(),
            'requirements': self._requirements_template(),
            'package_json': self._package_json_template(),
            'readme': self._readme_template(),
            'gitignore': self._gitignore_template(),
            'env': self._env_template(),
            'sql_schema': self._sql_schema_template(),
            'api_endpoint': self._api_endpoint_template()
        }
    
    def generate(self, file_type: str, name: str, **kwargs) -> GeneratedFile:
        """
        Generate a file of specified type
        
        Args:
            file_type: Type of file to generate
            name: Name for the file/component/class
            **kwargs: Additional parameters for template
            
        Returns:
            GeneratedFile object
        """
        if file_type not in self.templates:
            raise ValueError(f"Unknown file type: {file_type}. Available: {list(self.templates.keys())}")
        
        # Generate content based on type
        if file_type == 'python_script':
            content = self._generate_python_script(name, **kwargs)
            filename = f"{self._to_snake_case(name)}.py"
        elif file_type == 'python_class':
            content = self._generate_python_class(name, **kwargs)
            filename = f"{self._to_snake_case(name)}.py"
        elif file_type == 'python_test':
            content = self._generate_python_test(name, **kwargs)
            filename = f"test_{self._to_snake_case(name)}.py"
        elif file_type == 'react_component':
            content = self._generate_react_component(name, **kwargs)
            filename = f"{self._to_pascal_case(name)}.tsx"
        elif file_type == 'docker':
            content = self._generate_dockerfile(**kwargs)
            filename = "Dockerfile"
        elif file_type == 'docker_compose':
            content = self._generate_docker_compose(**kwargs)
            filename = "docker-compose.yml"
        elif file_type == 'readme':
            content = self._generate_readme(name, **kwargs)
            filename = "README.md"
        elif file_type == 'api_endpoint':
            content = self._generate_api_endpoint(name, **kwargs)
            filename = f"{self._to_snake_case(name)}_api.py"
        else:
            content = self.templates[file_type]
            filename = self._determine_filename(file_type, name)
        
        generated_file = GeneratedFile(
            filename=filename,
            content=content,
            file_type=file_type,
            metadata={'name': name, **kwargs}
        )
        
        self.generated_files.append(generated_file)
        return generated_file
    
    def _generate_python_script(self, name: str, description: str = "", 
                               imports: List[str] = None, main: bool = True) -> str:
        """Generate a Python script"""
        imports = imports or []
        import_str = '\n'.join(imports)
        date_str = datetime.now().strftime("%Y-%m-%d")
        
        template = f'''"""
{name}
{description or f"Generated Python script for {name}"}
Generated by ADAM on {date_str}
"""

{import_str}
import logging

logger = logging.getLogger(__name__)


def main():
    """Main function"""
    logger.info("Starting {name}")
    # TODO: Implement main logic
    pass


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
'''
        return template
    
    def _generate_python_class(self, name: str, base_class: str = None,
                              methods: List[str] = None) -> str:
        """Generate a Python class"""
        class_name = self._to_pascal_case(name)
        base = f"({base_class})" if base_class else ""
        methods = methods or ['__init__', '__str__']
        date_str = datetime.now().strftime("%Y-%m-%d")
        
        method_defs = []
        for method in methods:
            if method == '__init__':
                method_defs.append(f"""    def __init__(self):
        \"\"\"Initialize {class_name}\"\"\"
        pass""")
            elif method == '__str__':
                method_defs.append(f"""    def __str__(self):
        \"\"\"String representation\"\"\"
        return f"{class_name}()"
""")
            else:
                method_defs.append(f"""    def {method}(self):
        \"\"\"TODO: Implement {method}\"\"\"
        raise NotImplementedError""")
        
        template = f'''"""
{class_name} class
Generated by ADAM on {date_str}
"""

from typing import Any, Dict, List, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


class {class_name}{base}:
    """{class_name} implementation"""
    
{chr(10).join(method_defs)}
'''
        return template
    
    def _generate_python_test(self, name: str, test_framework: str = 'pytest') -> str:
        """Generate a Python test file"""
        module_name = self._to_snake_case(name)
        class_name = self._to_pascal_case(name)
        date_str = datetime.now().strftime("%Y-%m-%d")
        
        if test_framework == 'pytest':
            template = f'''"""
Tests for {module_name}
Generated by ADAM on {date_str}
"""

import pytest
from unittest.mock import Mock, patch
from {module_name} import {class_name}


class Test{class_name}:
    """Test cases for {class_name}"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.instance = {class_name}()
    
    def test_initialization(self):
        """Test {class_name} initialization"""
        assert self.instance is not None
    
    def test_basic_functionality(self):
        """Test basic functionality"""
        # TODO: Implement test
        pass
    
    @pytest.mark.parametrize("input,expected", [
        ("test1", "result1"),
        ("test2", "result2"),
    ])
    def test_parametrized(self, input, expected):
        """Test with parameters"""
        # TODO: Implement parametrized test
        pass
'''
        else:
            template = f'''"""
Tests for {module_name}
Generated by ADAM on {date_str}
"""

import unittest
from {module_name} import {class_name}


class Test{class_name}(unittest.TestCase):
    """Test cases for {class_name}"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.instance = {class_name}()
    
    def test_initialization(self):
        """Test {class_name} initialization"""
        self.assertIsNotNone(self.instance)
    
    def test_basic_functionality(self):
        """Test basic functionality"""
        # TODO: Implement test
        pass


if __name__ == '__main__':
    unittest.main()
'''
        return template
    
    def _generate_react_component(self, name: str, props: List[str] = None,
                                 hooks: List[str] = None, typescript: bool = True) -> str:
        """Generate a React component"""
        component_name = self._to_pascal_case(name)
        props = props or []
        hooks = hooks or ['useState']
        date_str = datetime.now().strftime("%Y-%m-%d")
        
        ext = 'tsx' if typescript else 'jsx'
        
        prop_interface = ""
        if typescript and props:
            prop_list = '\n  '.join([f"{prop}: string;" for prop in props])
            prop_interface = f"""
interface {component_name}Props {{
  {prop_list}
}}
"""
        
        hook_imports = []
        if 'useState' in hooks:
            hook_imports.append('useState')
        if 'useEffect' in hooks:
            hook_imports.append('useEffect')
        
        hooks_str = f", {{ {', '.join(hook_imports)} }}" if hook_imports else ""
        
        template = f'''/**
 * {component_name} Component
 * Generated by ADAM on {date_str}
 */

import React{hooks_str} from 'react';
import './styles/{component_name}.css';
{prop_interface}

export const {component_name}: React.FC{f"<{component_name}Props>" if typescript and props else ""} = ({{{', '.join(props)}}}) => {{
  const [state, setState] = useState<string>('');
  
  useEffect(() => {{
    // Component mount logic
  }}, []);
  
  return (
    <div className="{self._to_kebab_case(name)}">
      <h2>{component_name}</h2>
      {{/* TODO: Implement component */}}
    </div>
  );
}};

export default {component_name};
'''
        return template
    
    def _generate_dockerfile(self, base_image: str = 'python:3.11-slim',
                           workdir: str = '/app', **kwargs) -> str:
        """Generate a Dockerfile"""
        date_str = datetime.now().strftime("%Y-%m-%d")
        template = f'''# Generated by ADAM on {date_str}
FROM {base_image}

# Set working directory
WORKDIR {workdir}

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    gcc \\
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create non-root user
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser {workdir}
USER appuser

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Expose port
EXPOSE 8000

# Run application
CMD ["python", "main.py"]
'''
        return template
    
    def _generate_docker_compose(self, services: List[str] = None) -> str:
        """Generate docker-compose.yml"""
        services = services or ['app', 'db', 'redis']
        date_str = datetime.now().strftime("%Y-%m-%d")
        
        service_configs = {
            'app': '''  app:
    build: .
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql://user:pass@db:5432/dbname
      - REDIS_URL=redis://redis:6379
    depends_on:
      - db
      - redis
    volumes:
      - .:/app''',
            
            'db': '''  db:
    image: postgres:15
    environment:
      - POSTGRES_USER=user
      - POSTGRES_PASSWORD=pass
      - POSTGRES_DB=dbname
    volumes:
      - postgres_data:/var/lib/postgresql/data
    ports:
      - "5432:5432"''',
      
            'redis': '''  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data'''
        }
        
        services_str = '\n\n'.join([service_configs.get(s, '') for s in services if s in service_configs])
        
        template = f'''# Generated by ADAM on {date_str}
version: '3.8'

services:
{services_str}

volumes:
  postgres_data:
  redis_data:

networks:
  default:
    name: adam-network
'''
        return template
    
    def _generate_readme(self, project_name: str, description: str = "",
                        features: List[str] = None, **kwargs) -> str:
        """Generate README.md"""
        features = features or []
        features_str = '\n'.join([f"- {feature}" for feature in features])
        date_str = datetime.now().strftime("%Y-%m-%d")
        description_text = description or "A project generated by ADAM"
        features_text = features_str or "- Feature 1\n- Feature 2\n- Feature 3"
        
        template = f'''# {project_name}

{description_text}

## Features

{features_text}

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/{self._to_kebab_case(project_name)}.git
cd {self._to_kebab_case(project_name)}

# Install dependencies
pip install -r requirements.txt
```

## Usage

```python
from {self._to_snake_case(project_name)} import main

main()
```

## Development

```bash
# Run tests
pytest

# Run with Docker
docker-compose up
```

## Contributing

Pull requests are welcome. For major changes, please open an issue first.

## License

[MIT](https://choosealicense.com/licenses/mit/)

---
Generated by ADAM on {date_str}
'''
        return template
    
    def _generate_api_endpoint(self, name: str, method: str = 'GET',
                              auth_required: bool = True) -> str:
        """Generate FastAPI endpoint"""
        endpoint_name = self._to_snake_case(name)
        path_name = self._to_kebab_case(name)
        date_str = datetime.now().strftime("%Y-%m-%d")
        
        auth_dep = ", current_user: User = Depends(get_current_user)" if auth_required else ""
        
        template = f'''"""
API endpoint for {name}
Generated by ADAM on {date_str}
"""

from fastapi import APIRouter, Depends, HTTPException, Query
from typing import List, Optional
from pydantic import BaseModel
from datetime import datetime

router = APIRouter(prefix="/{path_name}", tags=["{name}"])


class {self._to_pascal_case(name)}Request(BaseModel):
    """Request model for {name}"""
    name: str
    description: Optional[str] = None
    

class {self._to_pascal_case(name)}Response(BaseModel):
    """Response model for {name}"""
    id: int
    name: str
    description: Optional[str]
    created_at: datetime


@router.get("/", response_model=List[{self._to_pascal_case(name)}Response])
async def get_{endpoint_name}_list(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=100){auth_dep}
):
    """Get list of {name}"""
    # TODO: Implement logic
    return []


@router.get("/{{{endpoint_name}_id}}", response_model={self._to_pascal_case(name)}Response)
async def get_{endpoint_name}(
    {endpoint_name}_id: int{auth_dep}
):
    """Get specific {name}"""
    # TODO: Implement logic
    raise HTTPException(status_code=404, detail="{name} not found")


@router.post("/", response_model={self._to_pascal_case(name)}Response)
async def create_{endpoint_name}(
    request: {self._to_pascal_case(name)}Request{auth_dep}
):
    """Create new {name}"""
    # TODO: Implement logic
    return {self._to_pascal_case(name)}Response(
        id=1,
        name=request.name,
        description=request.description,
        created_at=datetime.now()
    )


@router.put("/{{{endpoint_name}_id}}", response_model={self._to_pascal_case(name)}Response)
async def update_{endpoint_name}(
    {endpoint_name}_id: int,
    request: {self._to_pascal_case(name)}Request{auth_dep}
):
    """Update {name}"""
    # TODO: Implement logic
    raise HTTPException(status_code=404, detail="{name} not found")


@router.delete("/{{{endpoint_name}_id}}")
async def delete_{endpoint_name}(
    {endpoint_name}_id: int{auth_dep}
):
    """Delete {name}"""
    # TODO: Implement logic
    return {{"message": "{name} deleted successfully"}}
'''
        return template
    
    # Template methods (return empty strings, actual generation happens above)
    def _python_template(self) -> str: return ""
    def _python_class_template(self) -> str: return ""
    def _python_test_template(self) -> str: return ""
    def _javascript_module_template(self) -> str: return ""
    def _react_component_template(self) -> str: return ""
    def _dockerfile_template(self) -> str: return ""
    def _docker_compose_template(self) -> str: return ""
    def _requirements_template(self) -> str: return ""
    def _package_json_template(self) -> str: return ""
    def _readme_template(self) -> str: return ""
    def _gitignore_template(self) -> str: return ""
    def _env_template(self) -> str: return ""
    def _sql_schema_template(self) -> str: return ""
    def _api_endpoint_template(self) -> str: return ""
    
    # Utility methods
    def _to_snake_case(self, name: str) -> str:
        """Convert to snake_case"""
        s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
        return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()
    
    def _to_pascal_case(self, name: str) -> str:
        """Convert to PascalCase"""
        return ''.join(word.capitalize() for word in name.replace('-', '_').split('_'))
    
    def _to_kebab_case(self, name: str) -> str:
        """Convert to kebab-case"""
        return self._to_snake_case(name).replace('_', '-')
    
    def _determine_filename(self, file_type: str, name: str) -> str:
        """Determine filename based on type"""
        mappings = {
            'requirements': 'requirements.txt',
            'package_json': 'package.json',
            'gitignore': '.gitignore',
            'env': '.env.example',
            'sql_schema': f"{self._to_snake_case(name)}_schema.sql"
        }
        return mappings.get(file_type, f"{self._to_snake_case(name)}.txt")
    
    def list_templates(self) -> List[str]:
        """List available file templates"""
        return list(self.templates.keys())
    
    def get_generated_files(self) -> List[GeneratedFile]:
        """Get list of generated files"""
        return self.generated_files


# Convenience functions
def generate_python_project(name: str, include_tests: bool = True) -> List[GeneratedFile]:
    """Generate a complete Python project structure"""
    generator = FileGenerator()
    files = []
    
    # Main module
    files.append(generator.generate('python_class', name))
    
    # Tests
    if include_tests:
        files.append(generator.generate('python_test', name))
    
    # Configuration files
    files.append(generator.generate('readme', name))
    files.append(generator.generate('requirements', 'requirements'))
    files.append(generator.generate('gitignore', 'gitignore'))
    
    return files


def generate_react_app(name: str) -> List[GeneratedFile]:
    """Generate React app structure"""
    generator = FileGenerator()
    files = []
    
    # Components
    files.append(generator.generate('react_component', f"{name}App"))
    files.append(generator.generate('react_component', f"{name}Header"))
    files.append(generator.generate('react_component', f"{name}Main"))
    
    # Config
    files.append(generator.generate('package_json', name))
    files.append(generator.generate('readme', name))
    
    return files


if __name__ == "__main__":
    # Demo usage
    generator = FileGenerator()
    
    # Generate a Python class
    py_file = generator.generate('python_class', 'DataProcessor', 
                                 methods=['process', 'validate', 'save'])
    print(f"Generated: {py_file.filename}")
    print(py_file.content[:500])
    
    # Generate a React component
    react_file = generator.generate('react_component', 'UserDashboard',
                                   props=['userId', 'userName'],
                                   hooks=['useState', 'useEffect'])
    print(f"\nGenerated: {react_file.filename}")
    print(react_file.content[:500])
    
    # List available templates
    print(f"\nAvailable templates: {generator.list_templates()}")