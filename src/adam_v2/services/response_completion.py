"""
Response Completion Service for ADAM v2.0
Handles incomplete responses by requesting continuation
"""

import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class CompletionRequest:
    """Request to complete an incomplete response"""
    original_response: str
    model: str
    original_prompt: str
    max_attempts: int = 3
    attempt: int = 1

class ResponseCompletionService:
    """Service for completing incomplete AI responses"""
    
    def __init__(self, llm_client=None):
        self.llm_client = llm_client
        self.completion_prompts = {
            'default': "Continue from where you left off:",
            'code': "Please complete the code you were writing:",
            'sql': "Please complete the SQL query:",
            'json': "Please complete the JSON response:",
            'explanation': "Please continue your explanation:"
        }
    
    async def complete_response(
        self,
        incomplete_response: str,
        original_prompt: str,
        model: str,
        response_type: str = 'default',
        max_tokens: Optional[int] = None
    ) -> str:
        """
        Complete an incomplete response by requesting continuation
        """
        if not self.llm_client:
            logger.error("No LLM client available for completion")
            return incomplete_response
        
        # Determine the type of completion needed
        completion_type = self._determine_completion_type(incomplete_response, response_type)
        
        # Build continuation prompt
        continuation_prompt = self._build_continuation_prompt(
            incomplete_response,
            original_prompt,
            completion_type
        )
        
        try:
            # Request continuation with the same model
            logger.info(f"Requesting completion from {model} for {completion_type} response")
            
            # Use a higher max_tokens for completion
            completion_max_tokens = max_tokens or 2000
            
            response = await self.llm_client.complete(
                prompt=continuation_prompt,
                model=model,
                temperature=0.3,  # Lower temperature for consistent completion
                max_tokens=completion_max_tokens,
                system_prompt="You are continuing a previous response that was cut off. Continue naturally from where it ended, maintaining the same style and format. Do not repeat what was already said."
            )
            
            # Combine the responses
            completed_response = self._combine_responses(
                incomplete_response,
                response.content,
                completion_type
            )
            
            logger.info(f"Successfully completed response, added {len(response.content)} chars")
            return completed_response
            
        except Exception as e:
            logger.error(f"Failed to complete response: {e}")
            return incomplete_response + "\n\n*[Unable to complete response]*"
    
    def _determine_completion_type(self, response: str, hint: str = 'default') -> str:
        """Determine what type of completion is needed"""
        
        if hint != 'default':
            return hint
        
        # Check for code blocks
        if '```' in response:
            # Check if unclosed
            if response.count('```') % 2 != 0:
                # Determine language
                if '```sql' in response.lower():
                    return 'sql'
                elif '```json' in response.lower():
                    return 'json'
                else:
                    return 'code'
        
        # Check for JSON
        if response.strip().startswith('{') or response.strip().startswith('['):
            return 'json'
        
        # Check for SQL keywords
        sql_keywords = ['SELECT', 'INSERT', 'UPDATE', 'DELETE', 'CREATE', 'WITH']
        if any(keyword in response.upper() for keyword in sql_keywords):
            return 'sql'
        
        # Default to explanation if it seems like prose
        if len(response) > 100 and '.' in response:
            return 'explanation'
        
        return 'default'
    
    def _build_continuation_prompt(
        self,
        incomplete_response: str,
        original_prompt: str,
        completion_type: str
    ) -> str:
        """Build a prompt to continue the incomplete response"""
        
        # Get the last 500 characters for context
        context_snippet = incomplete_response[-500:] if len(incomplete_response) > 500 else incomplete_response
        
        # Build the continuation prompt
        prompt = f"""Original request: {original_prompt[:500]}...

You started responding with:
...{context_snippet}

{self.completion_prompts.get(completion_type, self.completion_prompts['default'])}"""
        
        # Add specific instructions based on type
        if completion_type == 'code':
            prompt += "\n\nMake sure to close any open code blocks with ```"
        elif completion_type == 'json':
            prompt += "\n\nEnsure the JSON is properly closed with matching brackets/braces."
        elif completion_type == 'sql':
            prompt += "\n\nComplete the SQL query with proper syntax."
        
        return prompt
    
    def _combine_responses(
        self,
        original: str,
        continuation: str,
        completion_type: str
    ) -> str:
        """Intelligently combine the original and continuation responses"""
        
        # Remove any repeated content at the boundary
        overlap = self._find_overlap(original, continuation)
        
        if overlap:
            # Remove the overlapping part from continuation
            continuation = continuation[len(overlap):]
        
        # Handle specific types
        if completion_type in ['code', 'sql']:
            # Ensure proper spacing
            if not original.endswith('\n') and continuation and not continuation.startswith('\n'):
                return original + '\n' + continuation
        
        # Default combination
        return original + continuation
    
    def _find_overlap(self, text1: str, text2: str, min_overlap: int = 10) -> str:
        """Find overlapping text between end of text1 and beginning of text2"""
        
        max_overlap = min(len(text1), len(text2), 100)  # Check up to 100 chars
        
        for i in range(max_overlap, min_overlap - 1, -1):
            if text1[-i:] == text2[:i]:
                return text2[:i]
        
        return ""
    
    async def validate_completion(
        self,
        response: str,
        expected_format: str = None
    ) -> Dict[str, Any]:
        """Validate if a response is complete"""
        
        validation = {
            'is_complete': True,
            'issues': [],
            'suggested_action': None
        }
        
        # Check for unclosed delimiters
        if response.count('```') % 2 != 0:
            validation['is_complete'] = False
            validation['issues'].append('Unclosed code block')
            validation['suggested_action'] = 'complete_code'
        
        if response.count('(') != response.count(')'):
            validation['is_complete'] = False
            validation['issues'].append('Unbalanced parentheses')
        
        if response.count('[') != response.count(']'):
            validation['is_complete'] = False
            validation['issues'].append('Unbalanced brackets')
        
        if response.count('{') != response.count('}'):
            validation['is_complete'] = False
            validation['issues'].append('Unbalanced braces')
        
        # Check for sentence completion
        if response and not response.rstrip().endswith(('.', '!', '?', '```', '}', ']', ')')):
            # Check if it ends mid-word
            last_word = response.split()[-1] if response.split() else ""
            if last_word and last_word[-1].isalpha():
                validation['is_complete'] = False
                validation['issues'].append('Response ends mid-sentence')
                validation['suggested_action'] = 'complete_explanation'
        
        # Format-specific validation
        if expected_format == 'json':
            try:
                import json
                json.loads(response)
            except:
                validation['is_complete'] = False
                validation['issues'].append('Invalid JSON format')
                validation['suggested_action'] = 'complete_json'
        
        return validation