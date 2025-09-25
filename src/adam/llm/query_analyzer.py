"""
Query Analyzer for Intelligent Model Routing
Analyzes query complexity and selects the appropriate model
"""
import re
from typing import Dict, List, Optional, Tuple
from enum import Enum

class QueryComplexity(Enum):
    """Query complexity levels"""
    LOW = "low"        # Simple questions, memory recaps
    MEDIUM = "medium"  # Standard analysis, general queries  
    HIGH = "high"      # Code generation, deep thinking

class QueryAnalyzer:
    """Analyzes queries to determine complexity and best model"""
    
    def __init__(self):
        # Keywords indicating complex queries requiring reasoning
        self.complex_indicators = {
            'code_generation': [
                'write code', 'write a', 'create a function', 'implement', 'develop',
                'build a script', 'code for', 'program', 'class', 'algorithm',
                'refactor', 'optimize code', 'debug', 'fix this code',
                'write a function', 'write a class', 'write a script',
                'python function', 'async/await', 'rate limiting', 'error handling',
                'queue of tasks', 'process a queue'
            ],
            'deep_analysis': [
                'analyze deeply', 'explain in detail', 'comprehensive analysis',
                'deep dive', 'thorough investigation', 'root cause', 'why does',
                'how does this work', 'explain the logic', 'understand deeply'
            ],
            'complex_reasoning': [
                'reason about', 'think through', 'step by step', 'solve this problem',
                'design a solution', 'architect', 'plan for', 'strategy for',
                'pros and cons', 'trade-offs',  # Removed 'compare and contrast'  
                'distributed system', 'system architecture', 'scalability',
                '1M+ users', 'concurrent users', 'optimization for', 'performance tuning',
                'suggest optimizations', 'complex analysis', 'comprehensive analysis'
            ],
            'mathematical': [
                'calculate', 'derive', 'prove', 'mathematical', 'equation',
                'formula', 'theorem', 'statistics', 'probability'
            ]
        }
        
        # Keywords indicating medium complexity
        self.medium_indicators = {
            'general_analysis': [
                'analyze', 'explain', 'describe', 'summarize', 'review',
                'evaluate', 'assess', 'investigate', 'examine',
                'explain in detail', 'detailed explanation', 'how does',
                'neural networks', 'machine learning', 'deep learning',
                'what is', 'what are', 'difference between', 'compare'
            ],
            'queries': [
                'what is', 'how to', 'when should', 'best practices',
                'recommendations', 'suggestions', 'tips for', 'guide',
                'explain how', 'how does this work', 'detailed explanation'
            ],
            'data_tasks': [
                'query optimization', 'performance', 'bigquery', 'sql',
                'database', 'data processing', 'etl', 'pipeline',
                'pdt', 'dbt', 'looker', 'lookml'  # Add data tooling keywords
            ]
        }
        
        # Keywords indicating low complexity
        self.low_indicators = {
            'memory_tasks': [
                'recall', 'remember', 'what did we', 'previous conversation',
                'earlier discussion', 'memory recap', 'history', 'past',
                'what were those', 'bring the', 'list those', 'show me those'
            ],
            'simple_questions': [
                'define', 'what does', 'list', 'name', 'yes or no',
                'true or false', 'quick answer', 'brief', 'short',
                'ok adam', 'got it', 'i see', 'thanks', 'explain this image',
                'what\'s in this image', 'can you explain', 'help me understand',
                'can you help me', 'will you help', 'i need your help'  # Common helper phrases
            ],
            'basic_tasks': [
                'count', 'simple', 'basic', 'straightforward', 'easy',
                'quick check', 'status', 'current', 'latest',
                'difference between', 'what is the', 'what are the',
                'which table', 'tell me which', 'should i request'  # Basic identification tasks
            ]
        }
        
        # Length thresholds
        self.length_thresholds = {
            'very_long': 500,   # Very long queries often need deep analysis
            'long': 200,        # Long queries might need reasoning
            'medium': 100,      # Medium length
            'short': 50         # Short queries
        }
    
    def analyze_query(self, query: str) -> Tuple[QueryComplexity, Dict[str, any]]:
        """
        Analyze a query and return its complexity level with reasoning
        
        Returns:
            Tuple of (QueryComplexity, analysis_details)
        """
        query_lower = query.lower()
        analysis = {
            'length': len(query),
            'indicators_found': [],
            'confidence': 0.0,
            'reasoning': []
        }
        
        # Check for complex indicators
        complex_score = 0
        for category, keywords in self.complex_indicators.items():
            for keyword in keywords:
                if keyword in query_lower:
                    complex_score += 2
                    analysis['indicators_found'].append(f"complex:{category}:{keyword}")
                    analysis['reasoning'].append(f"Found complex indicator '{keyword}' in {category}")
        
        # Check for medium indicators
        medium_score = 0
        for category, keywords in self.medium_indicators.items():
            for keyword in keywords:
                if keyword in query_lower:
                    medium_score += 1
                    analysis['indicators_found'].append(f"medium:{category}:{keyword}")
        
        # Check for low indicators
        low_score = 0
        for category, keywords in self.low_indicators.items():
            for keyword in keywords:
                if keyword in query_lower:
                    low_score += 1
                    analysis['indicators_found'].append(f"low:{category}:{keyword}")
        
        # Length analysis with reduced weighting - length alone doesn't mean complexity
        if analysis['length'] > self.length_thresholds['very_long']:
            complex_score += 1  # Reduced weight - was 2
            analysis['reasoning'].append("Very long query")
        elif analysis['length'] > self.length_thresholds['long']:
            medium_score += 1  # Changed to medium score instead of complex
            analysis['reasoning'].append("Long query")
        elif analysis['length'] < self.length_thresholds['short']:
            low_score += 1
            analysis['reasoning'].append("Short query suggests simple task")
        
        # Code detection
        if self._contains_code(query):
            complex_score += 3
            analysis['reasoning'].append("Query contains code blocks")
        
        # Technical terms that suggest complexity
        technical_terms = [
            'distributed', 'concurrent', 'async', 'await', 'architecture',
            'scalability', 'optimization', 'performance', 'algorithm',
            'microservices', 'api design', 'system design'
        ]
        tech_count = sum(1 for term in technical_terms if term in query_lower)
        if tech_count >= 2:
            complex_score += 2
            analysis['reasoning'].append(f"Multiple technical terms detected ({tech_count})")
        elif tech_count >= 1:
            complex_score += 1
            analysis['reasoning'].append("Technical terminology detected")
        
        # Determine complexity with VERY conservative logic
        # Prefer LOW complexity when in doubt to use faster models
        # HIGH complexity should be VERY rare - only for explicit code generation
        if complex_score >= 8:  # Even higher threshold - need strong evidence
            complexity = QueryComplexity.HIGH
            analysis['confidence'] = min(complex_score / 10, 1.0)
        elif complex_score >= 4 and medium_score < low_score:  # Need clear complex signals
            complexity = QueryComplexity.MEDIUM
            analysis['confidence'] = min((medium_score + complex_score) / 8, 1.0)
        elif medium_score >= 5 and low_score < 3:  # Clear medium signals without low indicators
            complexity = QueryComplexity.MEDIUM  
            analysis['confidence'] = min(medium_score / 7, 1.0)
        else:
            # Default to LOW for speed - handles most conversational queries
            complexity = QueryComplexity.LOW
            analysis['confidence'] = min((low_score + 3) / 5, 1.0)
        
        analysis['complexity'] = complexity.value
        analysis['scores'] = {
            'complex': complex_score,
            'medium': medium_score,
            'low': low_score
        }
        
        return complexity, analysis
    
    def _contains_code(self, text: str) -> bool:
        """Check if text contains code blocks or code-like patterns"""
        code_patterns = [
            r'```[\s\S]*```',           # Markdown code blocks
            r'def\s+\w+\s*\(',          # Python functions
            r'class\s+\w+',             # Class definitions
            r'function\s+\w+\s*\(',     # JavaScript functions
            r'SELECT\s+.*\s+FROM',      # SQL queries
            r'import\s+\w+',            # Import statements
            r'\{[\s\S]*\}',             # Code blocks with braces
            r'if\s*\(.*\)\s*\{',        # If statements
            r'for\s*\(.*\)\s*\{',       # For loops
            r'async\s+def',             # Async functions
            r'await\s+\w+',             # Await calls
            r'try:\s*\n.*except',       # Error handling
        ]
        
        for pattern in code_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return True
        return False
    
    def recommend_model(self, complexity: QueryComplexity, available_models: List[str], query: str = "") -> str:
        """
        Recommend the best model based on complexity and availability
        
        Args:
            complexity: Query complexity level
            available_models: List of available model names
            query: Optional query text for more specific routing
            
        Returns:
            Recommended model name
        """
        # Check if this is specifically a code generation task
        query_lower = query.lower() if query else ""
        is_code_task = any(
            keyword in query_lower 
            for keyword in self.complex_indicators.get('code_generation', [])
        ) or self._contains_code(query)
        
        # Model preferences by complexity - Fast Grok models for everything
        if complexity == QueryComplexity.HIGH:
            if is_code_task:
                # Use grok-4-fast-reasoning for code generation tasks
                model_preferences = ['grok-4-fast-reasoning', 'grok-4-reasoning', 'grok-4', 'o4-mini-high']
            else:
                # Use grok-4-fast-reasoning for complex reasoning tasks
                model_preferences = ['grok-4-fast-reasoning', 'grok-4-reasoning', 'grok-4', 'o4-mini-high']
        elif complexity == QueryComplexity.MEDIUM:
            # Use fast non-reasoning for medium queries
            model_preferences = ['grok-4-fast-non-reasoning', 'grok-4', 'gpt-4', 'gpt-4o-mini']
        else:  # LOW complexity
            # Use fast non-reasoning for basic queries
            model_preferences = ['grok-4-fast-non-reasoning', 'gpt-4.1-mini-2025-04-14', 'gpt-4o-mini', 'gpt-3.5-turbo', 'grok-3-mini']
        
        # Find first available model from preferences
        for preferred_model in model_preferences:
            if preferred_model in available_models:
                return preferred_model
        
        # Fallback to any available model
        return available_models[0] if available_models else None
    
    def get_reasoning_effort(self, complexity: QueryComplexity) -> Optional[str]:
        """
        Get appropriate reasoning effort level based on complexity
        
        Args:
            complexity: Query complexity level
            
        Returns:
            Reasoning effort level (low, medium, high) or None
        """
        effort_mapping = {
            QueryComplexity.LOW: "low",
            QueryComplexity.MEDIUM: "medium", 
            QueryComplexity.HIGH: "high"
        }
        return effort_mapping.get(complexity)


def analyze_and_route(query: str, available_models: List[str]) -> Tuple[str, Dict]:
    """
    Convenience function to analyze query and get model recommendation
    
    Args:
        query: The user query
        available_models: List of available model names
        
    Returns:
        Tuple of (recommended_model, analysis_details)
    """
    analyzer = QueryAnalyzer()
    complexity, analysis = analyzer.analyze_query(query)
    recommended_model = analyzer.recommend_model(complexity, available_models, query)
    reasoning_effort = analyzer.get_reasoning_effort(complexity)
    
    analysis['recommended_model'] = recommended_model
    analysis['reasoning_effort'] = reasoning_effort
    
    return recommended_model, analysis


# Example usage
if __name__ == "__main__":
    test_queries = [
        "What did we discuss yesterday?",
        "Write a Python function to process BigQuery data",
        "Explain how neural networks work in detail",
        "List the files in the current directory",
        "Design a scalable architecture for a real-time data processing system",
        "Quick summary of our last conversation"
    ]
    
    analyzer = QueryAnalyzer()
    available_models = ['grok-4-reasoning', 'grok-4', 'grok-3-mini-high']
    
    for query in test_queries:
        print(f"\nQuery: {query[:50]}...")
        complexity, analysis = analyzer.analyze_query(query)
        model = analyzer.recommend_model(complexity, available_models, query)
        print(f"Complexity: {complexity.value}")
        print(f"Recommended model: {model}")
        print(f"Confidence: {analysis['confidence']:.2f}")
        print(f"Reasoning: {analysis['reasoning'][:2]}")