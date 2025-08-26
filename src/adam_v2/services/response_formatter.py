"""
Response Formatter Service for ADAM v2.0
Handles formatting, validation, and fixing of AI responses
"""

import re
import json
import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
import textwrap

logger = logging.getLogger(__name__)

@dataclass
class FormattedResponse:
    """Formatted AI response with metadata"""
    content: str
    original_content: str
    model: str
    was_truncated: bool = False
    was_reformatted: bool = False
    formatting_issues: List[str] = None
    code_blocks: List[Dict[str, str]] = None

class ResponseFormatter:
    """Service for formatting and fixing AI responses"""
    
    def __init__(self):
        self.max_retries = 3
        self.min_response_length = 50  # Minimum acceptable response length
        self.code_indent_size = 2  # Default indent size for code
        
    async def format_response(
        self,
        content: str,
        model: str,
        context: Optional[Dict[str, Any]] = None
    ) -> FormattedResponse:
        """
        Format and validate AI response
        Fixes common issues like bad indentation and incomplete responses
        """
        if not content:
            return FormattedResponse(
                content="",
                original_content="",
                model=model,
                was_truncated=True,
                formatting_issues=["Empty response"]
            )
        
        original_content = content
        issues = []
        code_blocks = []
        
        # Check for incomplete response patterns
        is_incomplete = self._detect_incomplete_response(content)
        if is_incomplete:
            issues.append("Incomplete response detected")
            logger.warning(f"Incomplete response from {model}: {len(content)} chars")
        
        # Extract and fix code blocks
        content, code_blocks = self._extract_and_fix_code_blocks(content, model)
        
        # Fix general indentation issues
        if model in ["grok-4-reasoning", "grok-4", "grok-3-mini-high"]:
            content = self._fix_grok_formatting(content)
            issues.append(f"Fixed {model} formatting issues")
        
        # Fix markdown formatting
        content = self._fix_markdown_formatting(content)
        
        # Handle truncated responses
        if self._is_truncated(content):
            content = self._complete_truncated_response(content, context)
            issues.append("Response was truncated")
        
        # Reinsert fixed code blocks
        content = self._reinsert_code_blocks(content, code_blocks)
        
        return FormattedResponse(
            content=content,
            original_content=original_content,
            model=model,
            was_truncated=is_incomplete or self._is_truncated(original_content),
            was_reformatted=content != original_content,
            formatting_issues=issues if issues else None,
            code_blocks=code_blocks if code_blocks else None
        )
    
    def _detect_incomplete_response(self, content: str) -> bool:
        """Detect if response is incomplete"""
        
        # Check for mid-sentence endings
        incomplete_patterns = [
            r'\.\.\.$',  # Ends with ellipsis
            r'[^.!?]\s*$',  # Doesn't end with punctuation
            r'\b(and|or|but|with|for|to|the|a|an)\s*$',  # Ends with connector word
            r'```[^`]*$',  # Unclosed code block
            r'\([^)]*$',  # Unclosed parenthesis
            r'\[[^\]]*$',  # Unclosed bracket
            r'{[^}]*$',  # Unclosed brace
            r'"[^"]*$',  # Unclosed quote
        ]
        
        # Check if content is too short
        if len(content) < self.min_response_length:
            return True
        
        # Check for incomplete patterns
        for pattern in incomplete_patterns:
            if re.search(pattern, content.strip(), re.IGNORECASE):
                return True
        
        # Check for unbalanced delimiters
        if content.count('```') % 2 != 0:
            return True
        if content.count('(') != content.count(')'):
            return True
        if content.count('[') != content.count(']'):
            return True
        if content.count('{') != content.count('}'):
            return True
        
        return False
    
    def _extract_and_fix_code_blocks(self, content: str, model: str) -> tuple:
        """Extract code blocks and fix their formatting"""
        code_blocks = []
        placeholder_template = "___CODE_BLOCK_{}___"
        
        # Find all code blocks
        code_pattern = r'```(\w*)\n?([\s\S]*?)```'
        matches = re.finditer(code_pattern, content)
        
        for i, match in enumerate(matches):
            language = match.group(1) or ""
            code = match.group(2)
            
            # Fix code indentation for all models
            code = self._fix_code_indentation(code, language)
            
            code_blocks.append({
                'index': i,
                'language': language,
                'original': match.group(0),
                'fixed_code': code,
                'placeholder': placeholder_template.format(i)
            })
        
        # Replace code blocks with placeholders
        for block in code_blocks:
            content = content.replace(block['original'], block['placeholder'])
        
        return content, code_blocks
    
    def _fix_code_indentation(self, code: str, language: str) -> str:
        """Fix code indentation issues"""
        if not code.strip():
            return code
        
        # First, fix token splitting issues (GPT-5 specific)
        code = self._fix_token_splitting(code)
        
        lines = code.split('\n')
        
        # Detect if SQL (special handling for SQL formatting)
        if language.lower() in ['sql', 'postgresql', 'mysql']:
            return self._fix_sql_indentation(lines)
        
        # Detect if Python (use 4 spaces)
        if language.lower() in ['python', 'py']:
            return self._fix_python_indentation(lines)
        
        # For other languages, use smart indentation
        return self._fix_general_indentation(lines)
    
    def _fix_token_splitting(self, code: str) -> str:
        """Fix tokens that were split into individual characters"""
        
        # Fix lines that have excessive single-character splitting
        lines = code.split('\n')
        fixed_lines = []
        
        for line in lines:
            # Check if line has excessive splitting (single chars with spaces)
            if re.search(r'(\b\w\s+){3,}', line):
                # Join single characters that are separated by spaces
                # But preserve legitimate spacing around operators
                parts = []
                tokens = line.split()
                i = 0
                while i < len(tokens):
                    token = tokens[i]
                    # Check if this is a single character that should be joined
                    if len(token) == 1 and token.isalnum() and i + 1 < len(tokens):
                        # Look ahead to see if we should join
                        word = token
                        j = i + 1
                        while j < len(tokens) and len(tokens[j]) == 1 and tokens[j].isalnum():
                            word += tokens[j]
                            j += 1
                        if len(word) > 1:
                            parts.append(word)
                            i = j
                            continue
                    parts.append(token)
                    i += 1
                line = ' '.join(parts)
            
            fixed_lines.append(line)
        
        code = '\n'.join(fixed_lines)
        
        # Fix specific SQL keywords
        sql_keywords = [
            'SELECT', 'FROM', 'WHERE', 'WITH', 'INSERT', 'UPDATE', 'DELETE',
            'ORDER BY', 'GROUP BY', 'PARTITION BY', 'INNER JOIN', 'LEFT JOIN',
            'RIGHT JOIN', 'FULL JOIN', 'CROSS JOIN', 'HAVING', 'LIMIT',
            'DISTINCT', 'COUNT', 'SUM', 'AVG', 'MAX', 'MIN', 'CASE', 'WHEN',
            'THEN', 'ELSE', 'END', 'AND', 'OR', 'NOT', 'AS', 'ON', 'OVER'
        ]
        
        for keyword in sql_keywords:
            # Create pattern for split keyword
            split_pattern = r'\b' + r'\s+'.join(keyword) + r'\b'
            code = re.sub(split_pattern, keyword, code, flags=re.IGNORECASE)
        
        # Fix single character tokens with dots (e.g., "pp . *" -> "pp.*")
        code = re.sub(r'(\w+)\s+\.\s+(\w+|\*)', r'\1.\2', code)
        
        # Fix function calls split (e.g., "COUNT (" -> "COUNT(")
        code = re.sub(r'(\w+)\s+\(', r'\1(', code)
        
        # Fix operators
        code = re.sub(r'([=<>!])\s+([=])', r'\1\2', code)
        code = re.sub(r'(\w)\s+([,;])', r'\1\2', code)
        code = re.sub(r'=\s+(\w)', r'= \1', code)  # Preserve space after equals
        
        # Fix common patterns like "def name (" -> "def name("
        code = re.sub(r'\b(def|function|class|if|for|while)\s+(\w+)\s+\(', r'\1 \2(', code)
        
        return code
    
    def _fix_sql_indentation(self, lines: List[str]) -> str:
        """Fix SQL-specific indentation"""
        fixed_lines = []
        indent_level = 0
        in_cte = False
        in_parentheses = 0
        
        sql_keywords = {
            'cte': ['WITH'],
            'main_clause': ['SELECT', 'INSERT', 'UPDATE', 'DELETE', 'CREATE', 'DROP', 'ALTER'],
            'sub_clause': ['FROM', 'WHERE', 'GROUP BY', 'ORDER BY', 'HAVING', 'JOIN', 'LEFT JOIN', 'RIGHT JOIN', 'INNER JOIN', 'OUTER JOIN'],
            'connector': ['AND', 'OR', 'ON', 'AS', 'WHEN', 'THEN', 'ELSE'],
            'modifier': ['DISTINCT', 'ALL', 'TOP', 'LIMIT'],
            'function': ['COUNT', 'SUM', 'AVG', 'MIN', 'MAX', 'CASE', 'COALESCE', 'CAST'],
        }
        
        for line in lines:
            stripped = line.strip()
            if not stripped:
                fixed_lines.append('')
                continue
            
            # Count parentheses in current line
            open_parens = stripped.count('(')
            close_parens = stripped.count(')')
            
            # Adjust indent for closing parentheses at start of line
            if stripped.startswith(')'):
                in_parentheses = max(0, in_parentheses - 1)
                indent_level = max(0, indent_level - 1)
            
            # Determine proper indentation
            current_indent = indent_level
            if in_parentheses > 0:
                current_indent += in_parentheses
            
            # Apply indentation
            if current_indent > 0:
                fixed_line = '    ' * current_indent + stripped  # Use 4 spaces for SQL
            else:
                fixed_line = stripped
            
            fixed_lines.append(fixed_line)
            
            # Update parentheses count
            in_parentheses += open_parens - close_parens
            in_parentheses = max(0, in_parentheses)
            
            # Check for CTE start
            if stripped.upper().startswith('WITH'):
                in_cte = True
                indent_level = 1
            
            # Check for main query after CTE
            first_word = stripped.split()[0].upper() if stripped.split() else ''
            if first_word in sql_keywords['main_clause'] and not in_cte:
                indent_level = 0
            elif first_word in sql_keywords['main_clause'] and in_cte and in_parentheses == 0:
                in_cte = False
                indent_level = 0
            
            # Indent sub-clauses
            if first_word in sql_keywords['sub_clause']:
                if not in_cte or in_parentheses > 0:
                    indent_level = 1 if in_parentheses == 0 else indent_level
        
        return '\n'.join(fixed_lines)
    
    def _fix_python_indentation(self, lines: List[str]) -> str:
        """Fix Python-specific indentation"""
        fixed_lines = []
        indent_level = 0
        
        for line in lines:
            stripped = line.strip()
            if not stripped:
                fixed_lines.append('')
                continue
            
            # Decrease indent for dedent keywords
            if stripped.startswith(('else:', 'elif ', 'except:', 'finally:', 'except ')):
                indent_level = max(0, indent_level - 1)
            
            # Apply indentation
            if indent_level > 0:
                fixed_line = '    ' * indent_level + stripped  # 4 spaces for Python
            else:
                fixed_line = stripped
            
            fixed_lines.append(fixed_line)
            
            # Increase indent after colon
            if stripped.endswith(':'):
                indent_level += 1
            
            # Handle return, break, continue, pass
            if stripped.startswith(('return', 'break', 'continue', 'pass')):
                indent_level = max(0, indent_level - 1)
        
        return '\n'.join(fixed_lines)
    
    def _fix_general_indentation(self, lines: List[str]) -> str:
        """Fix general code indentation"""
        if not lines:
            return ''
        
        # Find minimum indentation (excluding empty lines)
        min_indent = float('inf')
        for line in lines:
            if line.strip():
                indent = len(line) - len(line.lstrip())
                min_indent = min(min_indent, indent)
        
        if min_indent == float('inf'):
            return '\n'.join(lines)
        
        # Remove minimum indentation from all lines
        fixed_lines = []
        for line in lines:
            if line.strip():
                fixed_lines.append(line[min_indent:])
            else:
                fixed_lines.append('')
        
        return '\n'.join(fixed_lines)
    
    def _fix_grok_formatting(self, content: str) -> str:
        """Fix Grok-specific formatting issues"""
        
        # Fix tokens split on individual characters (common GPT-5 issue)
        # Fix patterns like "s e l e c t" -> "select"
        content = re.sub(r'\b(\w)\s+(?=\w\b)', r'\1', content)
        
        # Fix operators split with spaces
        content = re.sub(r'([=<>!+\-*/])\s+([=<>])', r'\1\2', content)
        content = re.sub(r'(\w)\s+([.,;:])', r'\1\2', content)
        content = re.sub(r'([({])\s+', r'\1', content)
        content = re.sub(r'\s+([)}])', r'\1', content)
        
        # Fix weird spacing around punctuation
        content = re.sub(r'\s+([,.!?;:])', r'\1', content)
        content = re.sub(r'([,.!?;:])\s{2,}', r'\1 ', content)
        
        # Fix bullet point formatting
        content = re.sub(r'^(\s*)[-*]\s*', r'\1• ', content, flags=re.MULTILINE)
        
        # Fix numbered list formatting
        content = re.sub(r'^(\s*)(\d+)\.\s*', r'\1\2. ', content, flags=re.MULTILINE)
        
        # Fix excessive newlines
        content = re.sub(r'\n{3,}', '\n\n', content)
        
        return content
    
    def _fix_markdown_formatting(self, content: str) -> str:
        """Fix markdown formatting issues"""
        
        # Fix header spacing
        content = re.sub(r'^(#{1,6})([^#\s])', r'\1 \2', content, flags=re.MULTILINE)
        
        # Fix list item spacing
        content = re.sub(r'^(\s*[-*+])\s{2,}', r'\1 ', content, flags=re.MULTILINE)
        
        # Ensure blank lines around code blocks
        content = re.sub(r'([^\n])\n```', r'\1\n\n```', content)
        content = re.sub(r'```\n([^\n])', r'```\n\n\1', content)
        
        return content
    
    def _is_truncated(self, content: str) -> bool:
        """Check if response appears truncated"""
        
        truncation_indicators = [
            '...',
            '[truncated]',
            '[continued]',
            'etc.',
            'and so on'
        ]
        
        # Check last 50 characters for truncation indicators
        last_part = content[-50:].lower() if len(content) > 50 else content.lower()
        
        for indicator in truncation_indicators:
            if indicator in last_part:
                return True
        
        # Check if ends mid-code
        if '```' in content and content.count('```') % 2 != 0:
            return True
        
        return False
    
    def _complete_truncated_response(self, content: str, context: Optional[Dict[str, Any]]) -> str:
        """Attempt to complete truncated response"""
        
        # Add completion indicator
        if not content.endswith('.'):
            content += '...'
        
        # Close any open code blocks
        if content.count('```') % 2 != 0:
            content += '\n```'
        
        # Add truncation notice
        if context and context.get('show_truncation_notice', True):
            content += '\n\n*[Response was truncated. Please ask for continuation if needed.]*'
        
        return content
    
    def _reinsert_code_blocks(self, content: str, code_blocks: List[Dict]) -> str:
        """Reinsert fixed code blocks into content"""
        
        for block in code_blocks:
            # Recreate the code block with fixed formatting
            if block['language']:
                fixed_block = f"```{block['language']}\n{block['fixed_code']}\n```"
            else:
                fixed_block = f"```\n{block['fixed_code']}\n```"
            
            content = content.replace(block['placeholder'], fixed_block)
        
        return content
    
    async def validate_and_fix_response(
        self,
        response: str,
        expected_format: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Validate response format and attempt to fix if needed
        Useful for ensuring responses match expected formats (JSON, SQL, etc.)
        """
        
        validation_result = {
            'valid': False,
            'original': response,
            'fixed': None,
            'errors': [],
            'format': expected_format
        }
        
        if not response:
            validation_result['errors'].append('Empty response')
            return validation_result
        
        if expected_format == 'json':
            return self._validate_json_response(response)
        elif expected_format == 'sql':
            return self._validate_sql_response(response)
        elif expected_format == 'code':
            return self._validate_code_response(response)
        else:
            # General validation
            validation_result['valid'] = len(response) >= self.min_response_length
            validation_result['fixed'] = response
            if not validation_result['valid']:
                validation_result['errors'].append('Response too short')
        
        return validation_result
    
    def _validate_json_response(self, response: str) -> Dict[str, Any]:
        """Validate and fix JSON responses"""
        
        result = {
            'valid': False,
            'original': response,
            'fixed': None,
            'errors': [],
            'format': 'json'
        }
        
        # Try to extract JSON from response
        json_str = response
        
        # Remove markdown code blocks if present
        if '```json' in response:
            json_str = re.search(r'```json\n?(.*?)```', response, re.DOTALL)
            if json_str:
                json_str = json_str.group(1)
        elif '```' in response:
            json_str = re.search(r'```\n?(.*?)```', response, re.DOTALL)
            if json_str:
                json_str = json_str.group(1)
        
        # Try to parse JSON
        try:
            parsed = json.loads(json_str)
            result['valid'] = True
            result['fixed'] = json.dumps(parsed, indent=2)
            result['parsed'] = parsed
        except json.JSONDecodeError as e:
            result['errors'].append(f"JSON parse error: {str(e)}")
            
            # Try to fix common JSON issues
            fixed_json = json_str
            
            # Fix trailing commas
            fixed_json = re.sub(r',\s*}', '}', fixed_json)
            fixed_json = re.sub(r',\s*]', ']', fixed_json)
            
            # Fix single quotes
            fixed_json = re.sub(r"'([^']*)'", r'"\1"', fixed_json)
            
            # Try parsing again
            try:
                parsed = json.loads(fixed_json)
                result['valid'] = True
                result['fixed'] = json.dumps(parsed, indent=2)
                result['parsed'] = parsed
                result['errors'].append("Fixed JSON formatting issues")
            except:
                result['fixed'] = json_str
        
        return result
    
    def _validate_sql_response(self, response: str) -> Dict[str, Any]:
        """Validate SQL responses"""
        
        result = {
            'valid': False,
            'original': response,
            'fixed': None,
            'errors': [],
            'format': 'sql'
        }
        
        # Extract SQL from markdown if needed
        sql_str = response
        if '```sql' in response:
            sql_match = re.search(r'```sql\n?(.*?)```', response, re.DOTALL)
            if sql_match:
                sql_str = sql_match.group(1)
        elif '```' in response:
            sql_match = re.search(r'```\n?(.*?)```', response, re.DOTALL)
            if sql_match:
                sql_str = sql_match.group(1)
        
        # Basic SQL validation
        sql_keywords = ['SELECT', 'INSERT', 'UPDATE', 'DELETE', 'CREATE', 'ALTER', 'DROP', 'WITH']
        
        sql_upper = sql_str.upper()
        has_sql_keyword = any(keyword in sql_upper for keyword in sql_keywords)
        
        if has_sql_keyword:
            result['valid'] = True
            # Fix SQL formatting
            result['fixed'] = self._fix_sql_indentation(sql_str.split('\n'))
        else:
            result['errors'].append('No valid SQL keywords found')
            result['fixed'] = sql_str
        
        return result
    
    def _validate_code_response(self, response: str) -> Dict[str, Any]:
        """Validate code responses"""
        
        result = {
            'valid': False,
            'original': response,
            'fixed': None,
            'errors': [],
            'format': 'code'
        }
        
        # Check if response contains code blocks
        has_code_blocks = '```' in response
        
        if has_code_blocks:
            result['valid'] = True
            # Extract and fix code blocks
            _, code_blocks = self._extract_and_fix_code_blocks(response, 'unknown')
            result['code_blocks'] = code_blocks
            result['fixed'] = response  # Already fixed in extraction
        else:
            # Check if response is raw code
            if any(indicator in response for indicator in ['def ', 'function ', 'class ', 'const ', 'var ', 'let ']):
                result['valid'] = True
                result['fixed'] = f"```\n{response}\n```"
            else:
                result['errors'].append('No code blocks or code patterns found')
                result['fixed'] = response
        
        return result