"""
Enhanced logging configuration for ADAM
Detailed logging with color coding and multiple outputs
"""

import logging
import logging.handlers
import sys
from datetime import datetime
import os
from typing import Dict, Optional

# Color codes for terminal output
class ColoredFormatter(logging.Formatter):
    """Custom formatter with colors for terminal output"""
    
    COLORS = {
        'DEBUG': '\033[36m',    # Cyan
        'INFO': '\033[32m',     # Green
        'WARNING': '\033[33m',  # Yellow
        'ERROR': '\033[31m',    # Red
        'CRITICAL': '\033[35m', # Magenta
    }
    
    RESET = '\033[0m'
    
    def format(self, record):
        # Add color to level name
        levelname = record.levelname
        if levelname in self.COLORS:
            record.levelname = f"{self.COLORS[levelname]}{levelname}{self.RESET}"
        
        # Add emoji indicators
        if '🚀' not in record.msg and '✅' not in record.msg and '❌' not in record.msg:
            if levelname == 'ERROR':
                record.msg = f"❌ {record.msg}"
            elif levelname == 'WARNING':
                record.msg = f"⚠️ {record.msg}"
            elif 'Agent' in record.msg or 'agent' in record.msg:
                record.msg = f"🤖 {record.msg}"
        
        return super().format(record)

def setup_logging(log_level=None):
    """Setup comprehensive logging for ADAM"""
    
    # Determine log level
    if log_level is None:
        log_level = os.getenv('LOG_LEVEL', 'INFO')
    
    # Create logs directory
    os.makedirs('logs', exist_ok=True)
    
    # Root logger configuration
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper()))
    
    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # 1. Console Handler with colors
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_formatter = ColoredFormatter(
        '[%(asctime)s] %(levelname)s [%(name)s:%(lineno)d] %(message)s',
        datefmt='%H:%M:%S'
    )
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)
    
    # 2. File Handler for all logs
    file_handler = logging.handlers.RotatingFileHandler(
        'logs/adam_all.log',
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5
    )
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
    )
    file_handler.setFormatter(file_formatter)
    root_logger.addHandler(file_handler)
    
    # 3. Agent-specific log file
    agent_handler = logging.handlers.RotatingFileHandler(
        'logs/agent_execution.log',
        maxBytes=10*1024*1024,
        backupCount=3
    )
    agent_handler.setLevel(logging.DEBUG)
    agent_handler.setFormatter(file_formatter)
    
    # Add to agent-related loggers
    for logger_name in ['agents', 'tasks', 'services.agent_service', 'routers.messages']:
        logger = logging.getLogger(logger_name)
        logger.addHandler(agent_handler)
        logger.setLevel(logging.DEBUG)
    
    # 4. Error log file
    error_handler = logging.handlers.RotatingFileHandler(
        'logs/errors.log',
        maxBytes=5*1024*1024,
        backupCount=3
    )
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(file_formatter)
    root_logger.addHandler(error_handler)
    
    # 5. JSON structured logs for monitoring
    try:
        import json
        
        class JSONFormatter(logging.Formatter):
            def format(self, record):
                log_obj = {
                    'timestamp': datetime.utcnow().isoformat(),
                    'level': record.levelname,
                    'logger': record.name,
                    'message': record.getMessage(),
                    'filename': record.filename,
                    'line': record.lineno,
                    'function': record.funcName
                }
                
                if hasattr(record, 'task_id'):
                    log_obj['task_id'] = record.task_id
                    
                if record.exc_info:
                    log_obj['exception'] = self.formatException(record.exc_info)
                    
                return json.dumps(log_obj)
        
        json_handler = logging.handlers.RotatingFileHandler(
            'logs/structured.json',
            maxBytes=10*1024*1024,
            backupCount=3
        )
        json_handler.setLevel(logging.DEBUG)
        json_handler.setFormatter(JSONFormatter())
        root_logger.addHandler(json_handler)
        
    except Exception as e:
        print(f"Could not setup JSON logging: {e}")
    
    # Configure specific loggers
    logging.getLogger('uvicorn').setLevel(logging.WARNING)
    logging.getLogger('fastapi').setLevel(logging.INFO)
    
    # Log startup
    root_logger.info("=" * 60)
    root_logger.info("🚀 ADAM Logging System Initialized")
    root_logger.info(f"📊 Log Level: {log_level}")
    root_logger.info(f"📁 Log Directory: {os.path.abspath('logs')}")
    root_logger.info("=" * 60)
    
    return root_logger

# Utility function for agent task logging
def log_agent_step(task_id: str, step: str, details: Optional[dict] = None):
    """Log an agent execution step"""
    logger = logging.getLogger('agents')
    extra = {'task_id': task_id}
    
    message = f"[Task {task_id[:8]}] {step}"
    if details:
        message += f" - {details}"
    
    logger.info(message, extra=extra)