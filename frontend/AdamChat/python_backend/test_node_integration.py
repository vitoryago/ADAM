#!/usr/bin/env python3
"""
Test Node.js ↔ Python integration via stdin/stdout
"""
import json
import sys
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def main():
    """Main function for handling Node.js communication"""
    print("ADAM Python Backend Ready", file=sys.stderr)
    
    while True:
        try:
            # Read JSON message from stdin
            line = sys.stdin.readline()
            if not line:
                break
                
            # Parse the JSON message
            message = json.loads(line.strip())
            
            # Handle different message types based on Node.js format
            if message.get('type') == 'QUERY':
                response = handle_query_message(message)
            elif message.get('type') == 'COST_SUMMARY':
                response = handle_cost_summary(message)
            elif message.get('type') == 'test':
                response = handle_test_request(message)
            else:
                response = {
                    'requestId': message.get('requestId'),
                    'type': 'error',
                    'error': f"Unknown message type: {message.get('type')}"
                }
            
            # Send response back to Node.js with proper format
            response_line = f"ADAM_RESPONSE:{json.dumps(response)}"
            print(response_line)
            sys.stdout.flush()
            
        except json.JSONDecodeError as e:
            error_response = {
                'type': 'error',
                'error': f"JSON decode error: {str(e)}"
            }
            error_line = f"ADAM_ERROR:{json.dumps(error_response)}"
            print(error_line)
            sys.stdout.flush()
            
        except Exception as e:
            error_response = {
                'type': 'error',
                'error': f"Unexpected error: {str(e)}"
            }
            error_line = f"ADAM_ERROR:{json.dumps(error_response)}"
            print(error_line)
            sys.stdout.flush()

def handle_query_message(message):
    """Handle a QUERY message from Node.js"""
    request_data = message.get('data', {})
    user_message = request_data.get('query', '')
    project_id = request_data.get('projectId', 'default')
    conversation_id = request_data.get('conversationId', 'default')
    
    # Simulate AI response using the available API keys
    response_text = f"Hello! I'm ADAM, your advanced AI assistant. You said: '{user_message}' in project {project_id}. I have access to Grok-4 and OpenAI models with your configured API keys and am ready to help with your tasks."
    
    return {
        'requestId': message.get('requestId'),
        'response': {
            'response': response_text,
            'cost': 0.001,
            'modelUsed': 'grok-3-mini-fast',
            'processingTime': 250,
            'memoryConfidence': 0.8,
            'sources': [],
            'conversationState': {
                'complexity': 'simple',
                'memoryFound': False,
                'shouldStore': True
            }
        }
    }

def handle_cost_summary(message):
    """Handle cost summary request"""
    return {
        'requestId': message.get('requestId'),
        'response': {
            'dailyCost': 0.05,
            'monthlyCost': 1.25,
            'queryCount': 42,
            'modelUsage': {
                'grok-4': 0.03,
                'grok-3-mini-fast': 0.02,
                'gpt-4o-mini': 0.00
            }
        }
    }

def handle_config_request(message):
    """Handle configuration request"""
    config = {
        'adam_name': os.getenv('ADAM_NAME', 'ADAM'),
        'adam_language': os.getenv('ADAM_LANGUAGE', 'en'),
        'embedding_model': os.getenv('ADAM_EMBEDDING_MODEL', 'all-mpnet-base-v2'),
        'has_openai_key': bool(os.getenv('OPENAI_API_KEY')),
        'has_xai_key': bool(os.getenv('XAI_API_KEY')),
        'available_models': ['grok-4', 'grok-3-mini-fast', 'gpt-4o-mini']
    }
    
    return {
        'type': 'config_response',
        'config': config
    }

def handle_test_request(message):
    """Handle test request"""
    return {
        'type': 'test_response',
        'status': 'success',
        'message': 'ADAM Python backend is working correctly!',
        'api_keys_loaded': {
            'openai': bool(os.getenv('OPENAI_API_KEY')),
            'xai': bool(os.getenv('XAI_API_KEY'))
        }
    }

if __name__ == "__main__":
    main()