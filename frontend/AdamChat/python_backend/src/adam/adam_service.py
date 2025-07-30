#!/usr/bin/env python3
"""
ADAM Service Bridge
Main service that bridges the Node.js web application with the Python ADAM backend.
Handles communication via stdin/stdout and integrates all ADAM components.
"""

import json
import sys
import os
import asyncio
import logging
from typing import Dict, Any, Optional
from datetime import datetime

# Import ADAM components
from .integrated_conversation_system import IntegratedConversationSystem
from .cost_monitor import CostMonitor
from .memory import AdvancedMemorySystem
from .config import ADAMConfig
from .errors import ADAMError, ErrorRecoveryService

class ADAMService:
    """Main service bridge between Node.js and Python ADAM backend"""
    
    def __init__(self):
        self.config = ADAMConfig()
        self.conversation_system = None
        self.cost_monitor = None
        self.memory_system = None
        self.error_recovery = ErrorRecoveryService()
        
        # Setup logging
        self.setup_logging()
        
        # Initialize components
        self.initialize_components()
        
        self.logger.info("ADAM Service initialized successfully")
    
    def setup_logging(self):
        """Setup logging configuration"""
        log_level = os.getenv('LOG_LEVEL', 'INFO').upper()
        
        logging.basicConfig(
            level=getattr(logging, log_level),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('adam_service.log'),
                logging.StreamHandler(sys.stderr)  # Use stderr to avoid interfering with stdout communication
            ]
        )
        
        self.logger = logging.getLogger('ADAMService')
    
    def initialize_components(self):
        """Initialize all ADAM components"""
        try:
            # Initialize cost monitoring
            self.cost_monitor = CostMonitor()
            
            # Initialize memory system
            self.memory_system = AdvancedMemorySystem()
            
            # Initialize conversation system with all components
            self.conversation_system = IntegratedConversationSystem(
                memory_system=self.memory_system,
                cost_monitor=self.cost_monitor,
                config=self.config
            )
            
            self.logger.info("All ADAM components initialized")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize ADAM components: {e}")
            raise
    
    async def handle_query(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle a query request from the Node.js application"""
        try:
            query = request_data.get('query', '')
            conversation_id = request_data.get('conversationId', 'default')
            project_id = request_data.get('projectId', 'default')
            user_id = request_data.get('userId', 'anonymous')
            context = request_data.get('context', {})
            
            # Process query through integrated conversation system
            response = await self.conversation_system.process_conversation_turn(
                user_input=query,
                conversation_id=conversation_id,
                project_id=project_id,
                user_id=user_id,
                context=context
            )
            
            # Get cost information
            cost_info = self.cost_monitor.get_current_costs()
            
            # Get memory confidence if available
            memory_confidence = getattr(response, 'memory_confidence', 0.0)
            
            return {
                'response': response.response if hasattr(response, 'response') else str(response),
                'cost': response.cost if hasattr(response, 'cost') else 0.0,
                'modelUsed': response.model_used if hasattr(response, 'model_used') else 'unknown',
                'processingTime': response.processing_time if hasattr(response, 'processing_time') else 0.0,
                'memoryConfidence': memory_confidence,
                'sources': response.sources if hasattr(response, 'sources') else [],
                'conversationState': {
                    'complexity': response.complexity if hasattr(response, 'complexity') else 'simple',
                    'memoryFound': response.memory_found if hasattr(response, 'memory_found') else False,
                    'shouldStore': response.should_store if hasattr(response, 'should_store') else True
                },
                'costInfo': {
                    'dailyTotal': cost_info.get('daily_total', 0.0),
                    'monthlyTotal': cost_info.get('monthly_total', 0.0),
                    'dailyLimit': cost_info.get('daily_limit', 1.0),
                    'monthlyLimit': cost_info.get('monthly_limit', 30.0)
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error processing query: {e}")
            
            # Use error recovery service
            recovery_response = await self.error_recovery.handle_error(e, request_data)
            
            return {
                'response': recovery_response.get('message', 'I encountered an error processing your request. Please try again.'),
                'cost': 0.0,
                'modelUsed': 'error-recovery',
                'processingTime': 0.0,
                'memoryConfidence': 0.0,
                'sources': [],
                'conversationState': {
                    'complexity': 'error',
                    'memoryFound': False,
                    'shouldStore': False
                },
                'error': str(e)
            }
    
    async def handle_cost_summary(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle cost summary request"""
        try:
            return self.cost_monitor.get_detailed_summary()
        except Exception as e:
            self.logger.error(f"Error getting cost summary: {e}")
            return {
                'error': str(e),
                'dailyTotal': 0.0,
                'monthlyTotal': 0.0,
                'dailyLimit': 1.0,
                'monthlyLimit': 30.0
            }
    
    async def handle_memory_info(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle memory information request"""
        try:
            project_id = request_data.get('projectId', 'default')
            return await self.memory_system.get_memory_summary(project_id)
        except Exception as e:
            self.logger.error(f"Error getting memory info: {e}")
            return {
                'error': str(e),
                'memoryCount': 0,
                'connections': 0,
                'lastUpdated': datetime.now().isoformat()
            }
    
    async def process_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Process incoming request and route to appropriate handler"""
        request_type = request.get('type', 'QUERY')
        request_id = request.get('requestId', 'unknown')
        data = request.get('data', {})
        
        self.logger.debug(f"Processing request {request_id} of type {request_type}")
        
        try:
            if request_type == 'QUERY':
                response_data = await self.handle_query(data)
            elif request_type == 'COST_SUMMARY':
                response_data = await self.handle_cost_summary(data)
            elif request_type == 'MEMORY_INFO':
                response_data = await self.handle_memory_info(data)
            else:
                raise ValueError(f"Unknown request type: {request_type}")
            
            return {
                'requestId': request_id,
                'response': response_data
            }
            
        except Exception as e:
            self.logger.error(f"Error processing request {request_id}: {e}")
            return {
                'requestId': request_id,
                'error': str(e)
            }
    
    async def run(self):
        """Main service loop - read from stdin, process, write to stdout"""
        self.logger.info("ADAM Service starting main loop")
        
        try:
            while True:
                # Read line from stdin
                line = sys.stdin.readline()
                if not line:
                    break
                
                line = line.strip()
                if not line:
                    continue
                
                try:
                    # Parse JSON request
                    request = json.loads(line)
                    
                    # Process request
                    response = await self.process_request(request)
                    
                    # Send JSON response to stdout
                    print(json.dumps(response), flush=True)
                    
                except json.JSONDecodeError as e:
                    self.logger.error(f"Invalid JSON received: {e}")
                    error_response = {
                        'requestId': 'unknown',
                        'error': f'Invalid JSON: {str(e)}'
                    }
                    print(json.dumps(error_response), flush=True)
                
                except Exception as e:
                    self.logger.error(f"Unexpected error: {e}")
                    error_response = {
                        'requestId': 'unknown',
                        'error': f'Service error: {str(e)}'
                    }
                    print(json.dumps(error_response), flush=True)
        
        except KeyboardInterrupt:
            self.logger.info("ADAM Service stopped by user")
        except Exception as e:
            self.logger.error(f"Fatal error in main loop: {e}")
        finally:
            await self.shutdown()
    
    async def shutdown(self):
        """Cleanup and shutdown the service"""
        self.logger.info("ADAM Service shutting down")
        
        try:
            if self.conversation_system:
                await self.conversation_system.shutdown()
            
            if self.cost_monitor:
                await self.cost_monitor.save_state()
            
            if self.memory_system:
                await self.memory_system.save_state()
                
        except Exception as e:
            self.logger.error(f"Error during shutdown: {e}")
        
        self.logger.info("ADAM Service shutdown complete")

async def main():
    """Main entry point"""
    service = ADAMService()
    await service.run()

if __name__ == '__main__':
    asyncio.run(main())