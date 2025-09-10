"""
AI-Powered SQL Generator for ADAM
Uses LLMs to generate intelligent, context-aware SQL queries
"""

import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
import json

logger = logging.getLogger(__name__)

try:
    from adam.llm.client import UnifiedLLMClient
    from adam.llm.config import LLMConfig
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False
    logger.warning("LLM client not available for AI SQL generation")


@dataclass
class SQLGenerationResult:
    """Result from AI SQL generation"""
    query: str
    explanation: str
    dialect: str  # snowflake, postgres, mysql, etc.
    optimizations: List[str] = None
    warnings: List[str] = None
    estimated_cost: Optional[str] = None
    
    def to_dict(self):
        return {
            'query': self.query,
            'explanation': self.explanation,
            'dialect': self.dialect,
            'optimizations': self.optimizations or [],
            'warnings': self.warnings or []
        }


class AISQLGenerator:
    """
    Generate SQL queries using LLMs based on natural language requests
    This is the RIGHT way to do it - let the AI understand and create queries
    """
    
    def __init__(self, dialect: str = 'snowflake'):
        """
        Initialize AI SQL Generator
        
        Args:
            dialect: SQL dialect (snowflake, postgres, mysql, etc.)
        """
        if not LLM_AVAILABLE:
            raise ImportError("LLM client required for AI SQL generation")
            
        self.client = UnifiedLLMClient()
        self.dialect = dialect
        self.schema_context = {}
        
    def set_schema_context(self, schema: Dict[str, Any]):
        """
        Provide schema information for better query generation
        
        Args:
            schema: Dictionary with table definitions, columns, relationships
        """
        self.schema_context = schema
        
    def generate_query(self, request: str, 
                      tables_hint: List[str] = None,
                      optimization_level: str = 'balanced',
                      include_explanation: bool = True) -> SQLGenerationResult:
        """
        Generate SQL query from natural language request
        
        Args:
            request: Natural language description of what the query should do
            tables_hint: Optional list of tables that might be relevant
            optimization_level: 'fast', 'balanced', or 'thorough'
            include_explanation: Whether to explain the query logic
            
        Returns:
            SQLGenerationResult with the generated query
        """
        # Build context prompt with schema information
        schema_info = ""
        if self.schema_context:
            schema_info = f"\n\nAvailable schema:\n{json.dumps(self.schema_context, indent=2)}"
        elif tables_hint:
            schema_info = f"\n\nRelevant tables: {', '.join(tables_hint)}"
        
        # Determine optimization instructions
        optimization_instructions = {
            'fast': "Prioritize query execution speed. Use appropriate indexes and avoid full table scans.",
            'balanced': "Balance between readability and performance. Use CTEs for clarity.",
            'thorough': "Optimize for completeness and accuracy. Include data quality checks."
        }.get(optimization_level, "")
        
        # Build the prompt for SQL generation
        system_prompt = f"""You are an expert {self.dialect.upper()} SQL developer.
Generate optimized SQL queries based on user requirements.
Always follow {self.dialect} best practices and syntax.
{optimization_instructions}
Include comments in the SQL for clarity."""

        generation_prompt = f"""Generate a {self.dialect.upper()} SQL query for the following requirement:

{request}
{schema_info}

Provide:
1. The SQL query with appropriate comments
2. Brief explanation of the approach
3. Any optimization suggestions
4. Potential warnings or considerations

Format the response as:
```sql
-- Your SQL query here
```

Explanation: [Your explanation]
Optimizations: [List any optimizations applied]
Warnings: [Any warnings or considerations]"""

        try:
            # Use appropriate model based on complexity
            if 'complex' in request.lower() or 'optimize' in request.lower():
                model = 'grok-4-reasoning'  # Use reasoning model for complex queries
            else:
                model = 'grok-4'  # Standard model for regular queries
            
            response = self.client.chat(
                prompt=generation_prompt,
                model=model,
                system_prompt=system_prompt,
                temperature=0.2  # Lower temperature for more consistent SQL
            )
            
            # Parse the response
            content = response.content
            
            # Extract SQL query
            import re
            sql_match = re.search(r'```sql\n(.*?)\n```', content, re.DOTALL)
            query = sql_match.group(1) if sql_match else content.split('\n\n')[0]
            
            # Extract explanation
            explanation = ""
            if 'Explanation:' in content:
                explanation = content.split('Explanation:')[1].split('\n')[0].strip()
            
            # Extract optimizations
            optimizations = []
            if 'Optimizations:' in content:
                opt_text = content.split('Optimizations:')[1].split('\n')[0]
                optimizations = [o.strip() for o in opt_text.split(',')]
            
            # Extract warnings
            warnings = []
            if 'Warnings:' in content:
                warn_text = content.split('Warnings:')[1].split('\n')[0]
                warnings = [w.strip() for w in warn_text.split(',')]
            
            return SQLGenerationResult(
                query=query,
                explanation=explanation,
                dialect=self.dialect,
                optimizations=optimizations,
                warnings=warnings
            )
            
        except Exception as e:
            logger.error(f"SQL generation failed: {e}")
            raise
    
    def optimize_query(self, existing_query: str, 
                      optimization_goals: List[str] = None) -> SQLGenerationResult:
        """
        Optimize an existing SQL query
        
        Args:
            existing_query: The SQL query to optimize
            optimization_goals: List of goals like ['reduce_cost', 'improve_speed', 'add_partitioning']
            
        Returns:
            Optimized query with explanation
        """
        goals = optimization_goals or ['improve_performance', 'reduce_cost']
        goals_text = ', '.join(goals)
        
        prompt = f"""Optimize this {self.dialect} SQL query with these goals: {goals_text}

Original query:
```sql
{existing_query}
```

Provide:
1. Optimized version of the query
2. Explanation of changes made
3. Expected performance improvements"""

        response = self.client.chat(
            prompt=prompt,
            model='grok-4-reasoning',  # Use reasoning for optimization
            temperature=0.3
        )
        
        # Parse response similar to generate_query
        content = response.content
        sql_match = re.search(r'```sql\n(.*?)\n```', content, re.DOTALL)
        optimized_query = sql_match.group(1) if sql_match else content.split('\n\n')[0]
        
        explanation = content.split('```')[2].strip() if '```' in content else content
        
        return SQLGenerationResult(
            query=optimized_query,
            explanation=explanation,
            dialect=self.dialect,
            optimizations=goals
        )
    
    def explain_query(self, query: str) -> str:
        """
        Explain what a SQL query does in plain English
        
        Args:
            query: SQL query to explain
            
        Returns:
            Plain English explanation
        """
        prompt = f"""Explain this SQL query in plain English:

```sql
{query}
```

Provide:
1. What the query does
2. Which tables/data it accesses
3. Any filters or conditions applied
4. The expected output"""

        response = self.client.chat(
            prompt=prompt,
            model='grok-3-mini-high',  # Simple model for explanations
            temperature=0.5
        )
        
        return response.content
    
    def generate_from_results(self, 
                            data_sample: Dict[str, Any],
                            analysis_request: str) -> SQLGenerationResult:
        """
        Generate a follow-up query based on previous results
        
        Args:
            data_sample: Sample of previous query results
            analysis_request: What to analyze next
            
        Returns:
            Follow-up query
        """
        prompt = f"""Based on these query results:

```json
{json.dumps(data_sample, indent=2)[:1000]}  # First 1000 chars
```

Generate a {self.dialect} SQL query to: {analysis_request}

Consider the data structure and types shown in the results."""

        response = self.client.chat(
            prompt=prompt,
            model='grok-4',
            temperature=0.3
        )
        
        # Parse response
        content = response.content
        sql_match = re.search(r'```sql\n(.*?)\n```', content, re.DOTALL)
        query = sql_match.group(1) if sql_match else content
        
        return SQLGenerationResult(
            query=query,
            explanation=f"Follow-up query for: {analysis_request}",
            dialect=self.dialect
        )
    
    def convert_dialect(self, query: str, 
                       from_dialect: str, 
                       to_dialect: str) -> SQLGenerationResult:
        """
        Convert SQL query from one dialect to another
        
        Args:
            query: Original SQL query
            from_dialect: Source dialect (e.g., 'postgres')
            to_dialect: Target dialect (e.g., 'snowflake')
            
        Returns:
            Converted query
        """
        prompt = f"""Convert this {from_dialect} SQL query to {to_dialect}:

```sql
{query}
```

Ensure the converted query:
1. Uses {to_dialect}-specific syntax and functions
2. Maintains the same logic and results
3. Takes advantage of {to_dialect} features where applicable"""

        response = self.client.chat(
            prompt=prompt,
            model='grok-4',
            temperature=0.2
        )
        
        content = response.content
        sql_match = re.search(r'```sql\n(.*?)\n```', content, re.DOTALL)
        converted_query = sql_match.group(1) if sql_match else content
        
        return SQLGenerationResult(
            query=converted_query,
            explanation=f"Converted from {from_dialect} to {to_dialect}",
            dialect=to_dialect
        )


class SnowflakeAISQLGenerator(AISQLGenerator):
    """
    Specialized generator for Snowflake with platform-specific optimizations
    """
    
    def __init__(self):
        super().__init__(dialect='snowflake')
        
    def generate_time_travel_query(self, 
                                  table: str,
                                  time_point: str,
                                  analysis: str) -> SQLGenerationResult:
        """
        Generate query using Snowflake's time travel feature
        """
        prompt = f"""Generate a Snowflake query using TIME TRAVEL to:
- Access table {table} at {time_point}
- Perform this analysis: {analysis}

Use AT(TIMESTAMP) or BEFORE(TIMESTAMP) syntax appropriately."""

        return self.generate_query(prompt, tables_hint=[table])
    
    def generate_semi_structured_query(self,
                                      table: str,
                                      json_column: str,
                                      extraction_request: str) -> SQLGenerationResult:
        """
        Generate query for semi-structured data (JSON/VARIANT)
        """
        prompt = f"""Generate a Snowflake query to:
- Extract data from JSON/VARIANT column {json_column} in table {table}
- {extraction_request}

Use Snowflake's semi-structured data functions like:
- FLATTEN() for arrays
- Path notation (column:path.to.field)
- GET_PATH() when needed"""

        return self.generate_query(prompt, tables_hint=[table])


# Convenience functions
def generate_sql(request: str, dialect: str = 'snowflake') -> str:
    """
    Quick function to generate SQL from natural language
    
    Example:
        query = generate_sql("Find all customers who spent over $1000 last month")
    """
    generator = AISQLGenerator(dialect=dialect)
    result = generator.generate_query(request)
    return result.query


def optimize_sql(query: str, dialect: str = 'snowflake') -> str:
    """
    Optimize an existing SQL query
    
    Example:
        optimized = optimize_sql(my_slow_query)
    """
    generator = AISQLGenerator(dialect=dialect)
    result = generator.optimize_query(query)
    return result.query


if __name__ == "__main__":
    # Demo the AI SQL generator
    print("AI SQL Generator Demo")
    print("=" * 50)
    
    generator = AISQLGenerator('snowflake')
    
    # Example 1: Natural language to SQL
    request = "Find the top 10 customers by total purchase amount in the last 30 days"
    result = generator.generate_query(request, tables_hint=['orders', 'customers'])
    
    print(f"\nRequest: {request}")
    print(f"\nGenerated SQL:\n{result.query}")
    print(f"\nExplanation: {result.explanation}")
    
    # Example 2: Query optimization
    slow_query = """
    SELECT * FROM orders o
    JOIN customers c ON o.customer_id = c.id
    WHERE o.order_date > '2024-01-01'
    """
    
    print("\n" + "=" * 50)
    print("Optimizing query...")
    optimized = generator.optimize_query(slow_query, ['reduce_data_scanned', 'add_filters'])
    print(f"\nOptimized SQL:\n{optimized.query}")
    
    print("\n" + "=" * 50)
    print("\nThis is the RIGHT approach:")
    print("• LLM understands the business request")
    print("• Generates specific, optimized queries")
    print("• Adapts to your schema and data")
    print("• Learns from context and feedback")