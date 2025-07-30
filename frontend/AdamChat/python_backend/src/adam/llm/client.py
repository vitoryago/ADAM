"""
Unified LLM Client for ADAM
Supports OpenAI, Anthropic, and xAI (Grok) models with intelligent routing
"""
import asyncio
import os
import json
from typing import Dict, Any, List, Optional, AsyncGenerator
from datetime import datetime

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

# For xAI, we'll use HTTP requests since xai-sdk might not be available
import requests

class UnifiedLLMClient:
    """Unified client for multiple LLM providers"""
    
    def __init__(self):
        self.openai_client = None
        self.anthropic_client = None
        self.xai_api_key = os.getenv('XAI_API_KEY')
        
        # Initialize available clients
        if OPENAI_AVAILABLE and os.getenv('OPENAI_API_KEY'):
            self.openai_client = openai.AsyncOpenAI(
                api_key=os.getenv('OPENAI_API_KEY')
            )
        
        if ANTHROPIC_AVAILABLE and os.getenv('ANTHROPIC_API_KEY'):
            self.anthropic_client = anthropic.AsyncAnthropic(
                api_key=os.getenv('ANTHROPIC_API_KEY')
            )
    
    async def generate_response(self, query: str, context: Dict[str, Any], 
                              model: str, project_id: str) -> Dict[str, Any]:
        """Generate response using the specified model"""
        
        # Build the conversation context
        messages = self._build_messages(query, context)
        
        try:
            if model.startswith('grok') and self.xai_api_key:
                return await self._call_grok(messages, model)
            elif model.startswith('gpt') and self.openai_client:
                return await self._call_openai(messages, model)
            elif model.startswith('claude') and self.anthropic_client:
                return await self._call_anthropic(messages, model)
            else:
                # Fallback to best available model
                return await self._fallback_response(query, context)
                
        except Exception as e:
            return {
                'content': f"I encountered an error generating a response: {str(e)}. Please try again.",
                'input_tokens': 0,
                'output_tokens': 0,
                'error': str(e)
            }
    
    def _build_messages(self, query: str, context: Dict[str, Any]) -> List[Dict[str, str]]:
        """Build message array for the LLM"""
        messages = []
        
        # System message with ADAM personality and context
        system_content = self._build_system_message(context)
        messages.append({"role": "system", "content": system_content})
        
        # Add conversation history if available
        history = context.get('conversation_history', [])
        for msg in history[-6:]:  # Last 6 messages for context
            if msg.get('role') in ['user', 'assistant']:
                messages.append({
                    "role": msg['role'],
                    "content": msg['content']
                })
        
        # Add current user query
        messages.append({"role": "user", "content": query})
        
        return messages
    
    def _build_system_message(self, context: Dict[str, Any]) -> str:
        """Build system message with ADAM context"""
        adam_name = os.getenv('ADAM_NAME', 'ADAM')
        adam_language = os.getenv('ADAM_LANGUAGE', 'en')
        
        system_msg = f"""You are {adam_name}, an advanced AI assistant with sophisticated capabilities.

Your core traits:
- Intelligent and helpful with deep analytical thinking
- Project-aware with memory of previous interactions
- Cost-conscious and efficient in your responses
- Professional yet personable communication style

"""
        
        # Add memory context if available
        memory_results = context.get('memory_results', {})
        if memory_results.get('memories'):
            system_msg += "Recent memories from this project:\n"
            for memory in memory_results['memories'][:3]:
                system_msg += f"- {memory.get('query', '')} -> {memory.get('response', '')[:100]}...\n"
            system_msg += "\n"
        
        # Add project memory if available
        project_memory = context.get('project_memory', '')
        if project_memory.strip():
            system_msg += f"Project context:\n{project_memory[:500]}...\n\n"
        
        # Add RAG context if available
        rag_results = context.get('rag_results', {})
        if rag_results.get('context'):
            system_msg += f"Relevant knowledge:\n{rag_results['context'][:300]}...\n\n"
        
        # Add query analysis context
        analysis = context.get('analysis', {})
        if analysis:
            complexity = analysis.get('complexity', 'simple')
            query_type = analysis.get('type', 'general')
            system_msg += f"Query analysis: {complexity} {query_type} query\n\n"
        
        system_msg += f"Respond in {adam_language} language. Be concise but thorough."
        
        return system_msg
    
    async def _call_grok(self, messages: List[Dict[str, str]], model: str) -> Dict[str, Any]:
        """Call xAI Grok models via HTTP API"""
        
        # Map model names to xAI API names
        model_mapping = {
            'grok-4': 'grok-beta',
            'grok-4-reasoning': 'grok-beta', 
            'grok-3-mini-fast': 'grok-beta'
        }
        
        api_model = model_mapping.get(model, 'grok-beta')
        
        headers = {
            'Authorization': f'Bearer {self.xai_api_key}',
            'Content-Type': 'application/json'
        }
        
        payload = {
            'messages': messages,
            'model': api_model,
            'stream': False,
            'temperature': 0.7
        }
        
        try:
            # Use requests for HTTP call (since xAI Python SDK might not be available)
            response = requests.post(
                'https://api.x.ai/v1/chat/completions',
                headers=headers,
                json=payload,
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                choice = result['choices'][0]
                usage = result.get('usage', {})
                
                return {
                    'content': choice['message']['content'],
                    'input_tokens': usage.get('prompt_tokens', 0),
                    'output_tokens': usage.get('completion_tokens', 0)
                }
            else:
                raise Exception(f"xAI API error: {response.status_code} - {response.text}")
                
        except requests.RequestException as e:
            raise Exception(f"xAI API request failed: {str(e)}")
    
    async def _call_openai(self, messages: List[Dict[str, str]], model: str) -> Dict[str, Any]:
        """Call OpenAI models"""
        
        # Map model names
        model_mapping = {
            'gpt-4o-mini': 'gpt-4o-mini',
            'gpt-4o': 'gpt-4o',
            'gpt-4': 'gpt-4-turbo'
        }
        
        api_model = model_mapping.get(model, 'gpt-4o-mini')
        
        response = await self.openai_client.chat.completions.create(
            model=api_model,
            messages=messages,
            temperature=0.7,
            max_tokens=1500
        )
        
        return {
            'content': response.choices[0].message.content,
            'input_tokens': response.usage.prompt_tokens,
            'output_tokens': response.usage.completion_tokens
        }
    
    async def _call_anthropic(self, messages: List[Dict[str, str]], model: str) -> Dict[str, Any]:
        """Call Anthropic Claude models"""
        
        # Extract system message and user messages
        system_msg = ""
        user_messages = []
        
        for msg in messages:
            if msg['role'] == 'system':
                system_msg = msg['content']
            else:
                user_messages.append(msg)
        
        # Map model names
        model_mapping = {
            'claude-3-haiku': 'claude-3-haiku-20240307',
            'claude-3-sonnet': 'claude-3-5-sonnet-20241022',
            'claude-3-opus': 'claude-3-opus-20240229'
        }
        
        api_model = model_mapping.get(model, 'claude-3-haiku-20240307')
        
        response = await self.anthropic_client.messages.create(
            model=api_model,
            max_tokens=1500,
            temperature=0.7,
            system=system_msg,
            messages=user_messages
        )
        
        return {
            'content': response.content[0].text,
            'input_tokens': response.usage.input_tokens,
            'output_tokens': response.usage.output_tokens
        }
    
    async def _fallback_response(self, query: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Fallback response when no models are available"""
        
        # Try to use any available model
        if self.openai_client:
            messages = self._build_messages(query, context)
            return await self._call_openai(messages, 'gpt-4o-mini')
        elif self.xai_api_key:
            messages = self._build_messages(query, context)
            return await self._call_grok(messages, 'grok-3-mini-fast')
        elif self.anthropic_client:
            messages = self._build_messages(query, context)
            return await self._call_anthropic(messages, 'claude-3-haiku')
        else:
            return {
                'content': f"I'm ADAM, your AI assistant. You asked: '{query}'. I'm currently experiencing connectivity issues with my AI models. Please ensure your API keys are properly configured.",
                'input_tokens': len(query.split()) * 2,
                'output_tokens': 30
            }