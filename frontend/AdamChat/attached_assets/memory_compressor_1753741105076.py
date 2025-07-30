"""
Memory Compression System for ADAM
Implements intelligent LLM-based compression for aging memories
"""
import asyncio
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import json
import re
from dataclasses import dataclass
import logging

# For LLM integration
try:
    from .llm.client import UnifiedLLMClient
    from .llm.config import LLMConfig
except ImportError:
    from llm.client import UnifiedLLMClient
    from llm.config import LLMConfig

logger = logging.getLogger(__name__)

@dataclass
class CompressionResult:
    """Result of memory compression"""
    compressed_content: str
    compression_ratio: float
    preserved_elements: List[str]  # What was kept
    removed_elements: List[str]    # What was removed
    compression_level: str
    tokens_saved: int

class MemoryCompressor:
    """
    Intelligent memory compression using LLMs
    Preserves semantic value while reducing storage
    """
    
    def __init__(self):
        self.llm_config = LLMConfig()
        self.llm_client = UnifiedLLMClient(self.llm_config)
        
        # Compression prompts for different levels
        self.compression_prompts = {
            "moderate": self._get_moderate_prompt,
            "high": self._get_high_prompt,
            "ultra": self._get_ultra_prompt
        }
    
    async def compress_memory(self, 
                            content: str, 
                            metadata: Dict,
                            compression_level: str) -> CompressionResult:
        """
        Compress a memory using LLM-based summarization
        """
        if compression_level not in self.compression_prompts:
            raise ValueError(f"Unknown compression level: {compression_level}")
        
        # Get appropriate prompt
        prompt = self.compression_prompts[compression_level](content, metadata)
        
        try:
            # Use grok-3-mini for compression (cost-effective)
            response = await self.llm_client.complete(
                prompt=prompt,
                model="grok-3-mini" if "grok-3-mini" in self.llm_config.get_available_models() else None,
                temperature=0.3  # Lower temperature for consistent compression
            )
            
            # Parse the compressed content
            compressed = self._parse_compression_response(response.content)
            
            # Calculate metrics
            original_length = len(content)
            compressed_length = len(compressed)
            compression_ratio = 1 - (compressed_length / original_length)
            
            # Estimate tokens saved (rough approximation)
            tokens_saved = (original_length - compressed_length) // 4
            
            # Extract what was preserved/removed
            preserved, removed = self._analyze_compression(content, compressed, compression_level)
            
            return CompressionResult(
                compressed_content=compressed,
                compression_ratio=compression_ratio,
                preserved_elements=preserved,
                removed_elements=removed,
                compression_level=compression_level,
                tokens_saved=tokens_saved
            )
            
        except Exception as e:
            logger.error(f"LLM compression failed: {e}")
            # Fallback to rule-based compression
            return await self._fallback_compression(content, metadata, compression_level)
    
    def _get_moderate_prompt(self, content: str, metadata: Dict) -> str:
        """Moderate compression: Keep key exchanges, remove redundancy"""
        memory_type = metadata.get('memory_type', 'general')
        
        return f"""You are a memory compression system. Compress this {memory_type} memory while preserving important information.

COMPRESSION RULES for MODERATE level (reduce by ~50%):
1. Keep: All questions, answers, code snippets, errors, solutions
2. Keep: Key insights and learning points
3. Remove: Pleasantries, redundant explanations, verbose descriptions
4. Remove: Repetitive content and filler words
5. Preserve: Technical accuracy and context

Original Memory:
{content}

Provide ONLY the compressed version. Do not add any explanation or metadata."""
    
    def _get_high_prompt(self, content: str, metadata: Dict) -> str:
        """High compression: Extract only key insights"""
        memory_type = metadata.get('memory_type', 'general')
        query_text = metadata.get('query_text', 'N/A')
        
        return f"""You are a memory compression system. Extract only the essential insights from this {memory_type} memory.

Original Query: {query_text}

COMPRESSION RULES for HIGH level (reduce by ~80%):
1. Extract: Core problem and final solution only
2. Include: Critical code snippets (max 3 lines)
3. Include: Key error messages and fixes
4. Format: "Problem: X | Solution: Y | Key insight: Z"
5. Maximum 3-5 sentences total

Original Memory:
{content}

Provide ONLY the compressed insights in the format specified."""
    
    def _get_ultra_prompt(self, content: str, metadata: Dict) -> str:
        """Ultra compression: Single paragraph summary"""
        memory_type = metadata.get('memory_type', 'general')
        topics = metadata.get('topics', [])
        
        return f"""You are a memory compression system. Create a single sentence summary of this {memory_type} memory.

Topics: {', '.join(topics) if topics else 'general'}

COMPRESSION RULES for ULTRA level (reduce by ~95%):
1. Single sentence, maximum 20 words
2. Include: What problem was solved or what was learned
3. Focus on the outcome, not the process
4. Use format: "[Action] [Object] [Result]"

Original Memory:
{content[:500]}...

Provide ONLY the one-sentence summary."""
    
    def _parse_compression_response(self, response: str) -> str:
        """Clean up the LLM response"""
        # Remove any markdown formatting
        compressed = re.sub(r'```[\s\S]*?```', lambda m: m.group(0), response)
        
        # Remove any explanatory text that might have been added
        lines = compressed.split('\n')
        cleaned_lines = []
        
        for line in lines:
            # Skip lines that look like metadata or explanations
            if any(skip in line.lower() for skip in ['compressed:', 'summary:', 'note:']):
                continue
            cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines).strip()
    
    def _analyze_compression(self, original: str, compressed: str, level: str) -> Tuple[List[str], List[str]]:
        """Analyze what was preserved and removed"""
        preserved = []
        removed = []
        
        # Check for preserved elements
        if '```' in compressed:
            preserved.append("code snippets")
        if 'error' in compressed.lower():
            preserved.append("error messages")
        if 'solution' in compressed.lower() or 'fixed' in compressed.lower():
            preserved.append("solutions")
        if '?' in compressed:
            preserved.append("questions")
        
        # Estimate removed elements based on level
        if level == "moderate":
            removed = ["pleasantries", "redundant explanations", "verbose descriptions"]
        elif level == "high":
            removed = ["detailed explanations", "intermediate steps", "context", "examples"]
        elif level == "ultra":
            removed = ["details", "code", "explanations", "context", "most content"]
        
        return preserved, removed
    
    async def _fallback_compression(self, content: str, metadata: Dict, level: str) -> CompressionResult:
        """Rule-based compression when LLM fails"""
        logger.info(f"Using fallback compression for level: {level}")
        
        lines = content.split('\n')
        
        if level == "moderate":
            # Keep lines with important keywords
            important_keywords = ['error', 'solution', 'query:', 'response:', 'def', 'class', 'import']
            compressed_lines = [
                line for line in lines 
                if any(keyword in line.lower() for keyword in important_keywords) or len(line.strip()) > 20
            ]
            compressed = '\n'.join(compressed_lines[:len(lines)//2])  # Keep max 50%
            
        elif level == "high":
            # Extract query/response pairs
            compressed_parts = []
            if metadata.get('query_text'):
                compressed_parts.append(f"Query: {metadata['query_text'][:100]}...")
            
            # Find solution patterns
            for i, line in enumerate(lines):
                if any(word in line.lower() for word in ['solution:', 'fixed:', 'answer:']):
                    compressed_parts.append(line)
                    if i + 1 < len(lines):
                        compressed_parts.append(lines[i + 1])
            
            compressed = '\n'.join(compressed_parts[:5])  # Max 5 lines
            
        else:  # ultra
            # Single line summary from metadata
            query = metadata.get('query_text', 'Query')[:50]
            memory_type = metadata.get('memory_type', 'Memory')
            compressed = f"{memory_type}: {query} (compressed from {len(content)} chars)"
        
        compression_ratio = 1 - (len(compressed) / len(content))
        
        return CompressionResult(
            compressed_content=compressed,
            compression_ratio=compression_ratio,
            preserved_elements=["basic structure"],
            removed_elements=["details"],
            compression_level=level,
            tokens_saved=(len(content) - len(compressed)) // 4
        )
    
    async def batch_compress(self, memories: List[Tuple[str, str, Dict, str]]) -> Dict[str, CompressionResult]:
        """
        Compress multiple memories in batch for efficiency
        Args: List of (memory_id, content, metadata, compression_level)
        """
        results = {}
        
        # Group by compression level for better prompting
        by_level = {}
        for memory_id, content, metadata, level in memories:
            if level not in by_level:
                by_level[level] = []
            by_level[level].append((memory_id, content, metadata))
        
        # Process each level
        for level, items in by_level.items():
            logger.info(f"Compressing {len(items)} memories at {level} level")
            
            # Process in parallel but with concurrency limit
            tasks = []
            for memory_id, content, metadata in items:
                task = self.compress_memory(content, metadata, level)
                tasks.append((memory_id, task))
            
            # Gather results with concurrency limit
            for memory_id, task in tasks:
                try:
                    result = await task
                    results[memory_id] = result
                except Exception as e:
                    logger.error(f"Failed to compress memory {memory_id}: {e}")
        
        return results
    
    def estimate_compression_savings(self, total_memories: int, memory_sizes: List[int]) -> Dict:
        """Estimate storage savings from compression"""
        avg_size = sum(memory_sizes) / len(memory_sizes) if memory_sizes else 1000
        
        # Estimate based on tier distribution
        active_ratio = 0.2      # 20% stay active
        moderate_ratio = 0.3    # 30% moderate compression
        high_ratio = 0.3        # 30% high compression
        ultra_ratio = 0.2       # 20% ultra compression
        
        # Compression ratios by level
        savings = {
            'active': 0,
            'moderate': 0.5,
            'high': 0.8,
            'ultra': 0.95
        }
        
        total_original = total_memories * avg_size
        total_compressed = (
            total_original * active_ratio * (1 - savings['active']) +
            total_original * moderate_ratio * (1 - savings['moderate']) +
            total_original * high_ratio * (1 - savings['high']) +
            total_original * ultra_ratio * (1 - savings['ultra'])
        )
        
        return {
            'original_size_mb': total_original / (1024 * 1024),
            'compressed_size_mb': total_compressed / (1024 * 1024),
            'savings_mb': (total_original - total_compressed) / (1024 * 1024),
            'savings_percent': (1 - total_compressed / total_original) * 100,
            'estimated_cost_savings_yearly': (total_original - total_compressed) / (1024 * 1024 * 1024) * 0.023 * 12  # S3 pricing estimate
        }


class CompressionValidator:
    """Validate that compression preserves essential information"""
    
    @staticmethod
    def validate_compression(original: str, compressed: str, level: str) -> Tuple[bool, List[str]]:
        """
        Validate that compression preserved essential elements
        Returns: (is_valid, issues)
        """
        issues = []
        
        # Check compression achieved
        if len(compressed) >= len(original):
            issues.append("No compression achieved")
        
        # Level-specific validation
        if level == "moderate":
            # Should preserve code and errors
            if '```' in original and '```' not in compressed:
                issues.append("Code blocks lost")
            if 'error' in original.lower() and 'error' not in compressed.lower():
                issues.append("Error information lost")
                
        elif level == "high":
            # Should be significantly shorter
            if len(compressed) > len(original) * 0.3:
                issues.append("Insufficient compression for high level")
                
        elif level == "ultra":
            # Should be a single paragraph
            if len(compressed.split('\n')) > 3:
                issues.append("Ultra compression should be 1-3 lines max")
        
        # Check for information preservation
        if not compressed.strip():
            issues.append("Compression resulted in empty content")
        
        return len(issues) == 0, issues


# Example usage and testing
async def demo_compression():
    """Demonstrate memory compression"""
    compressor = MemoryCompressor()
    
    # Example memory content
    test_memory = """
Query: How do I optimize this SQL query that's running slowly?

Response: I'll help you optimize your SQL query. Here are several strategies:

1. First, let's analyze the query execution plan:
```sql
EXPLAIN ANALYZE
SELECT * FROM orders o
JOIN customers c ON o.customer_id = c.id
WHERE o.order_date > '2024-01-01'
AND c.country = 'US';
```

2. Add appropriate indexes:
```sql
CREATE INDEX idx_orders_date ON orders(order_date);
CREATE INDEX idx_customers_country ON customers(country);
```

3. Consider using a covering index if you're selecting specific columns.

4. For large datasets, you might want to partition the orders table by date.

The main issue was likely the missing indexes on the WHERE clause columns. After adding these indexes, the query should run much faster.

Let me know if you need help with the specific query you're working with!
"""
    
    test_metadata = {
        'memory_type': 'code_pattern',
        'query_text': 'How do I optimize this SQL query that\'s running slowly?',
        'topics': ['sql', 'optimization', 'performance']
    }
    
    # Test different compression levels
    for level in ['moderate', 'high', 'ultra']:
        print(f"\n{'='*50}")
        print(f"Testing {level.upper()} compression:")
        print(f"{'='*50}")
        
        result = await compressor.compress_memory(test_memory, test_metadata, level)
        
        print(f"Original length: {len(test_memory)} chars")
        print(f"Compressed length: {len(result.compressed_content)} chars")
        print(f"Compression ratio: {result.compression_ratio:.1%}")
        print(f"Tokens saved: ~{result.tokens_saved}")
        print(f"\nCompressed content:")
        print("-" * 30)
        print(result.compressed_content)
        print("-" * 30)
        print(f"Preserved: {', '.join(result.preserved_elements)}")
        print(f"Removed: {', '.join(result.removed_elements)}")
        
        # Validate
        is_valid, issues = CompressionValidator.validate_compression(
            test_memory, result.compressed_content, level
        )
        print(f"Validation: {'✓ Passed' if is_valid else '✗ Failed'}")
        if issues:
            print(f"Issues: {', '.join(issues)}")


if __name__ == "__main__":
    # Run demo
    asyncio.run(demo_compression())