"""
Celery tasks for agent execution
Real background processing with full logging
"""

from celery import Task, current_task
from celery_app import app
import logging
import asyncio
from typing import Dict, Any
import json
import traceback
from datetime import datetime

logger = logging.getLogger(__name__)

class AgentTask(Task):
    """Base task with logging and error handling"""
    
    def on_failure(self, exc, task_id, args, kwargs, einfo):
        """Log task failures"""
        logger.error(f"Task {task_id} failed: {exc}")
        logger.error(f"Traceback: {einfo}")
    
    def on_success(self, retval, task_id, args, kwargs):
        """Log task success"""
        logger.info(f"Task {task_id} completed successfully")

@app.task(base=AgentTask, bind=True, name='agent.execute')
def execute_agent_task(self, request: str, workspace_path: str = None) -> Dict[str, Any]:
    """
    Execute an agent task in the background
    This runs in a separate Celery worker process
    """
    
    task_id = self.request.id
    logger.info(f"Starting agent task {task_id}: {request[:100]}...")
    
    # Update task state
    self.update_state(
        state='PROCESSING',
        meta={
            'current': 'Initializing agent',
            'request': request,
            'started_at': datetime.now().isoformat()
        }
    )
    
    try:
        # Import here to avoid circular imports
        from agents.agent_runtime import get_agent_runtime
        
        # Initialize runtime
        runtime = get_agent_runtime(workspace_path)
        
        # Log each step
        logger.info(f"Task {task_id}: Agent runtime initialized")
        
        self.update_state(
            state='PROCESSING',
            meta={'current': 'Processing request with agents'}
        )
        
        # Run the agent (convert async to sync for Celery)
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            result = loop.run_until_complete(
                runtime.process_request(request)
            )
        finally:
            loop.close()
        
        logger.info(f"Task {task_id}: Agent execution completed")
        
        # Return the result
        return {
            'task_id': task_id,
            'status': 'success',
            'result': result,
            'completed_at': datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Task {task_id} failed: {str(e)}")
        logger.error(traceback.format_exc())
        
        # Return error info
        return {
            'task_id': task_id,
            'status': 'error',
            'error': str(e),
            'traceback': traceback.format_exc(),
            'completed_at': datetime.now().isoformat()
        }

@app.task(base=AgentTask, bind=True, name='agent.explore_folder')
def explore_folder_task(self, folder_path: str) -> Dict[str, Any]:
    """
    Specific task for folder exploration
    """
    
    task_id = self.request.id
    logger.info(f"Exploring folder {folder_path} - Task {task_id}")
    
    self.update_state(
        state='PROCESSING',
        meta={'current': f'Exploring {folder_path}'}
    )
    
    try:
        # Import tools
        from adam.tools import ListFilesTool, ReadFileTool
        
        # Create tools
        list_tool = ListFilesTool()
        read_tool = ReadFileTool()
        
        # List files
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            # List directory
            list_result = loop.run_until_complete(
                list_tool.execute(directory=folder_path)
            )
            
            files = []
            if hasattr(list_result, 'data'):
                files = list_result.data
            
            # Analyze structure
            analysis = {
                'folder': folder_path,
                'total_files': len(files) if isinstance(files, list) else 0,
                'file_types': {},
                'subdirectories': [],
                'sql_models': [],
                'config_files': []
            }
            
            # Categorize files
            for file in files if isinstance(files, list) else []:
                if isinstance(file, dict):
                    name = file.get('name', '')
                else:
                    name = str(file)
                
                # Track file types
                if '.' in name:
                    ext = name.split('.')[-1]
                    analysis['file_types'][ext] = analysis['file_types'].get(ext, 0) + 1
                
                # Identify important files
                if name.endswith('.sql'):
                    analysis['sql_models'].append(name)
                elif name in ['dbt_project.yml', 'schema.yml', 'config.yml']:
                    analysis['config_files'].append(name)
                elif '/' in name:  # Subdirectory
                    analysis['subdirectories'].append(name)
            
            logger.info(f"Task {task_id}: Folder exploration completed")
            
            return {
                'task_id': task_id,
                'status': 'success',
                'analysis': analysis,
                'completed_at': datetime.now().isoformat()
            }
            
        finally:
            loop.close()
            
    except Exception as e:
        logger.error(f"Task {task_id} failed: {str(e)}")
        return {
            'task_id': task_id,
            'status': 'error',
            'error': str(e),
            'completed_at': datetime.now().isoformat()
        }

@app.task(base=AgentTask, bind=True, name='agent.create_file')
def create_file_task(self, file_path: str, content: str, base_file: str = None) -> Dict[str, Any]:
    """
    Task for creating files with optional base reference
    """
    
    task_id = self.request.id
    logger.info(f"Creating file {file_path} - Task {task_id}")
    
    self.update_state(
        state='PROCESSING',
        meta={'current': f'Creating {file_path}'}
    )
    
    try:
        from adam.tools import WriteFileTool, ReadFileTool
        
        write_tool = WriteFileTool()
        read_tool = ReadFileTool()
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            # If base file provided, read it first
            base_content = None
            if base_file:
                base_result = loop.run_until_complete(
                    read_tool.execute(file_path=base_file)
                )
                if hasattr(base_result, 'data'):
                    base_content = base_result.data
                    logger.info(f"Task {task_id}: Read base file {base_file}")
            
            # Generate content if needed
            if not content and base_content:
                # Use LLM to generate based on base
                from services.llm_service import LLMService
                llm = LLMService()
                
                prompt = f"Create a new file based on this template:\n{base_content[:1000]}"
                response = loop.run_until_complete(
                    llm.generate_response(
                        message=prompt,
                        history=[],
                        model="gpt-5-mini"
                    )
                )
                content = response.content
            
            # Write the file
            write_result = loop.run_until_complete(
                write_tool.execute(file_path=file_path, content=content)
            )
            
            logger.info(f"Task {task_id}: File created successfully")
            
            return {
                'task_id': task_id,
                'status': 'success',
                'file_path': file_path,
                'size': len(content),
                'completed_at': datetime.now().isoformat()
            }
            
        finally:
            loop.close()
            
    except Exception as e:
        logger.error(f"Task {task_id} failed: {str(e)}")
        return {
            'task_id': task_id,
            'status': 'error',
            'error': str(e),
            'completed_at': datetime.now().isoformat()
        }

# Task monitoring
@app.task(name='agent.get_status')
def get_task_status(task_id: str) -> Dict[str, Any]:
    """Get the status of a running task"""
    
    from celery.result import AsyncResult
    result = AsyncResult(task_id, app=app)
    
    return {
        'task_id': task_id,
        'state': result.state,
        'info': result.info,
        'ready': result.ready(),
        'successful': result.successful() if result.ready() else None,
        'result': result.result if result.ready() else None
    }