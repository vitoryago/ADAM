#!/usr/bin/env python3
"""
ADAM Main Service Entry Point
Production-ready ADAM AI assistant with full RAG capabilities
"""

import sys
import json
import os
import asyncio
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
from dotenv import load_dotenv

# Add ADAM src to Python path for imports
adam_root = Path(__file__).parent.parent.parent.parent  # Go up to ADAM root
sys.path.insert(0, str(adam_root / 'src'))

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=os.getenv('LOG_LEVEL', 'INFO'))
logger = logging.getLogger(__name__)

class ADAMService:
    """Main ADAM service for handling Node.js integration"""
    
    def __init__(self):
        self.initialized = False
        self.conversation_system = None
        self.project_manager = None
        self.cost_monitor = None
        
    async def initialize(self):
        """Initialize all ADAM components"""
        if self.initialized:
            return
            
        try:
            # Import ADAM modules
            from adam.integrated_conversation_system import IntegratedADAMSystem
            from adam.project_manager import ProjectManager
            from adam.cost_monitor import CostMonitor
            from adam.config import ADAMConfig
            from adam.memory import ADAMMemoryAdvanced

            # Initialize configuration
            config = ADAMConfig()

            # Initialize base memory system
            base_memory = ADAMMemoryAdvanced(
                persist_directory=str(config.memory_storage_path)
            )

            # Initialize components
            self.conversation_system = IntegratedADAMSystem(
                base_memory_system=base_memory,
                conversation_dir=str(config.conversation_storage_path)
            )
            self.project_manager = ProjectManager(base_path="./data/adam_projects")
            self.cost_monitor = CostMonitor(storage_path="./cost_tracking")
            
            # Components are already initialized
            
            self.initialized = True
            logger.info("ADAM service initialized successfully")
            print("ADAM Python Backend Ready", file=sys.stderr, flush=True)
            
        except Exception as e:
            logger.error(f"Failed to initialize ADAM service: {e}")
            print(f"ADAM initialization error: {e}", file=sys.stderr, flush=True)
            raise
    
    async def process_query(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process a user query through the full ADAM system"""
        try:
            query = request_data.get('query', '')
            project_id = request_data.get('projectId', 'default')
            conversation_id = request_data.get('conversationId', 'default')
            user_id = request_data.get('userId', 'anonymous')
            context = request_data.get('context', {})
            
            # Ensure project exists
            await self.project_manager.ensure_project(project_id)
            
            # Process through integrated conversation system
            result = await self.conversation_system.process_conversation(
                query=query,
                project_id=project_id,
                conversation_id=conversation_id,
                user_id=user_id,
                context=context
            )
            
            # Track costs and update project stats
            cost_info = await self.cost_monitor.track_query(
                model=result.get('model_used', 'unknown'),
                input_tokens=result.get('input_tokens', 0),
                output_tokens=result.get('output_tokens', 0)
            )
            
            # Update project statistics
            await self.project_manager.update_project_stats(
                project_id, cost_info.get('cost', 0.0)
            )
            
            return {
                'response': result.get('response', 'No response generated'),
                'cost': cost_info.get('cost', 0.0),
                'modelUsed': result.get('model_used', 'grok-3-mini-fast'),
                'processingTime': result.get('processing_time', 0),
                'memoryConfidence': result.get('memory_confidence', 0.0),
                'sources': result.get('sources', []),
                'conversationState': {
                    'complexity': result.get('complexity', 'simple'),
                    'memoryFound': result.get('memory_found', False),
                    'shouldStore': result.get('should_store', True)
                }
            }
            
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            return {
                'response': f"I apologize, but I encountered an error processing your request: {str(e)}",
                'cost': 0.0,
                'modelUsed': 'error',
                'processingTime': 0,
                'memoryConfidence': 0.0,
                'sources': [],
                'conversationState': {
                    'complexity': 'simple',
                    'memoryFound': False,
                    'shouldStore': False
                }
            }
    
    async def get_cost_summary(self) -> Dict[str, Any]:
        """Get cost summary from cost monitor"""
        try:
            return await self.cost_monitor.get_summary()
        except Exception as e:
            logger.error(f"Error getting cost summary: {e}")
            return {
                'dailyCost': 0.0,
                'monthlyCost': 0.0,
                'queryCount': 0,
                'modelUsage': {}
            }

async def main():
    """Main async function for handling Node.js communication"""
    adam_service = ADAMService()
    
    try:
        # Initialize ADAM service
        await adam_service.initialize()
        
        # Main communication loop
        while True:
            try:
                # Read JSON message from stdin
                line = sys.stdin.readline()
                if not line:
                    break
                    
                # Parse the JSON message
                message = json.loads(line.strip())
                
                # Handle different message types
                response = None
                if message.get('type') == 'QUERY':
                    request_data = message.get('data', {})
                    query_result = await adam_service.process_query(request_data)
                    response = {
                        'requestId': message.get('requestId'),
                        'response': query_result
                    }
                    
                elif message.get('type') == 'COST_SUMMARY':
                    cost_summary = await adam_service.get_cost_summary()
                    response = {
                        'requestId': message.get('requestId'),
                        'response': cost_summary
                    }
                    
                else:
                    response = {
                        'requestId': message.get('requestId'),
                        'error': f"Unknown message type: {message.get('type')}"
                    }
                
                # Send response back to Node.js
                if response:
                    response_line = f"ADAM_RESPONSE:{json.dumps(response)}"
                    print(response_line, flush=True)
                    
            except json.JSONDecodeError as e:
                error_response = {
                    'error': f"JSON decode error: {str(e)}"
                }
                error_line = f"ADAM_ERROR:{json.dumps(error_response)}"
                print(error_line, flush=True)
                
            except Exception as e:
                logger.error(f"Unexpected error in main loop: {e}")
                error_response = {
                    'error': f"Unexpected error: {str(e)}"
                }
                error_line = f"ADAM_ERROR:{json.dumps(error_response)}"
                print(error_line, flush=True)
                
    except KeyboardInterrupt:
        logger.info("ADAM service shutting down...")
    except Exception as e:
        logger.error(f"Fatal error in ADAM service: {e}")
        sys.exit(1)

if __name__ == "__main__":
    # Run the async main function
    asyncio.run(main())