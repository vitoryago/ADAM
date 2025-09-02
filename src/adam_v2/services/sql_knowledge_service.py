"""
Snowflake SQL Knowledge Service for ADAM v2
Provides intelligent SQL query optimization and formatting
"""

import yaml
import re
import os
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import logging
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class QueryType(Enum):
    SELECT = "select"
    INSERT = "insert"
    UPDATE = "update"
    DELETE = "delete"
    MERGE = "merge"
    CREATE = "create"
    ALTER = "alter"
    ANALYTICAL = "analytical"
    AGGREGATION = "aggregation"

@dataclass
class SQLOptimizationSuggestion:
    """SQL optimization suggestion"""
    issue: str
    suggestion: str
    severity: str  # 'critical', 'warning', 'info'
    example: Optional[str] = None

@dataclass
class SQLQueryRequest:
    """Request for SQL query generation"""
    query_type: QueryType
    tables: List[str]
    columns: Optional[List[str]] = None
    conditions: Optional[List[str]] = None
    joins: Optional[List[Dict]] = None
    aggregations: Optional[List[str]] = None
    order_by: Optional[List[str]] = None
    limit: Optional[int] = None

class SQLKnowledgeService:
    """Service for SQL query optimization and best practices"""
    
    def __init__(self):
        """Initialize the SQL knowledge service"""
        self.knowledge_base_path = Path(__file__).parent.parent.parent.parent / "knowledge"
        self.patterns_file = self.knowledge_base_path / "sql" / "snowflake_sql_patterns.yaml"
        self.knowledge_file = self.knowledge_base_path / "sql" / "SNOWFLAKE_SQL_KNOWLEDGE.md"
        
        # Load patterns and knowledge
        self.patterns = self._load_patterns()
        self.knowledge = self._load_knowledge()
        
        # SQL detection patterns
        self.sql_keywords = [
            'SELECT', 'FROM', 'WHERE', 'JOIN', 'GROUP BY', 'ORDER BY',
            'INSERT', 'UPDATE', 'DELETE', 'MERGE', 'CREATE', 'ALTER',
            'WITH', 'UNION', 'INTERSECT', 'EXCEPT'
        ]
        
        # Keywords that should be uppercase
        self.uppercase_keywords = set([
            'SELECT', 'FROM', 'WHERE', 'JOIN', 'INNER', 'LEFT', 'RIGHT', 'FULL', 'OUTER',
            'ON', 'AND', 'OR', 'NOT', 'IN', 'EXISTS', 'BETWEEN', 'LIKE', 'IS', 'NULL',
            'GROUP', 'BY', 'HAVING', 'ORDER', 'ASC', 'DESC', 'LIMIT', 'OFFSET',
            'UNION', 'ALL', 'INTERSECT', 'EXCEPT', 'WITH', 'AS', 'CASE', 'WHEN', 'THEN',
            'ELSE', 'END', 'DISTINCT', 'COUNT', 'SUM', 'AVG', 'MAX', 'MIN',
            'INSERT', 'INTO', 'VALUES', 'UPDATE', 'SET', 'DELETE', 'MERGE', 'USING',
            'CREATE', 'ALTER', 'DROP', 'TABLE', 'VIEW', 'INDEX', 'DATABASE', 'SCHEMA',
            'PARTITION', 'CLUSTER', 'PRIMARY', 'KEY', 'FOREIGN', 'REFERENCES',
            'CASCADE', 'RESTRICT', 'DEFAULT', 'CHECK', 'UNIQUE', 'CONSTRAINT',
            'GRANT', 'REVOKE', 'TO', 'ROLE', 'USER', 'PRIVILEGES',
            'BEGIN', 'COMMIT', 'ROLLBACK', 'TRANSACTION', 'IF', 'EXISTS',
            'TEMP', 'TEMPORARY', 'TRANSIENT', 'VOLATILE', 'RECURSIVE',
            'OVER', 'PARTITION', 'ROW_NUMBER', 'RANK', 'DENSE_RANK', 'NTILE',
            'LAG', 'LEAD', 'FIRST_VALUE', 'LAST_VALUE', 'LISTAGG',
            'PIVOT', 'UNPIVOT', 'LATERAL', 'FLATTEN', 'QUALIFY',
            'SAMPLE', 'TABLESAMPLE', 'CONNECT', 'START', 'PRIOR',
            'TRUE', 'FALSE', 'UNKNOWN', 'ANY', 'SOME', 'CAST', 'CONVERT',
            'COALESCE', 'NULLIF', 'NVL', 'DECODE', 'GREATEST', 'LEAST',
            'DATE', 'TIME', 'TIMESTAMP', 'INTERVAL', 'YEAR', 'MONTH', 'DAY',
            'HOUR', 'MINUTE', 'SECOND', 'CURRENT_DATE', 'CURRENT_TIME', 'CURRENT_TIMESTAMP',
            'DATEADD', 'DATEDIFF', 'DATE_TRUNC', 'EXTRACT', 'TO_DATE', 'TO_TIMESTAMP'
        ])
    
    def _load_patterns(self) -> Dict[str, Any]:
        """Load SQL patterns from YAML file"""
        if not self.patterns_file.exists():
            logger.warning(f"SQL patterns file not found at {self.patterns_file}")
            return {}
            
        try:
            with open(self.patterns_file, 'r') as f:
                patterns = yaml.safe_load(f)
                logger.info(f"Loaded SQL patterns from {self.patterns_file}")
                return patterns
        except Exception as e:
            logger.error(f"Error loading SQL patterns: {e}")
            return {}
    
    def _load_knowledge(self) -> str:
        """Load SQL knowledge from markdown file"""
        if not self.knowledge_file.exists():
            logger.warning(f"SQL knowledge file not found at {self.knowledge_file}")
            return ""
            
        try:
            with open(self.knowledge_file, 'r') as f:
                knowledge = f.read()
                logger.info(f"Loaded SQL knowledge from {self.knowledge_file}")
                return knowledge
        except Exception as e:
            logger.error(f"Error loading SQL knowledge: {e}")
            return ""
    
    def detect_sql_context(self, query: str) -> bool:
        """Detect if a query is SQL-related"""
        query_upper = query.upper()
        
        # Check for SQL keywords
        sql_indicators = [
            'SQL', 'QUERY', 'SELECT', 'TABLE', 'DATABASE',
            'SNOWFLAKE', 'OPTIMIZE', 'PERFORMANCE',
            'JOIN', 'AGGREGATE', 'INDEX', 'PARTITION'
        ]
        
        return any(indicator in query_upper for indicator in sql_indicators)
    
    async def intelligent_sql_detection(self, query: str) -> tuple[bool, float]:
        """Use LLM to intelligently detect if query needs SQL knowledge"""
        try:
            from adam.llm.client import UnifiedLLMClient
            
            detection_prompt = """Analyze if this query is asking about SQL queries, optimization, or database operations.

Query: "{query}"

SQL-related queries include:
- Writing or optimizing SQL queries
- Database performance tuning
- Query formatting and best practices
- Snowflake-specific SQL features
- Table design and indexing
- Aggregations, joins, window functions
- Data warehouse operations

Non-SQL queries include:
- General programming (unless database-related)
- Business logic (unless needs SQL implementation)
- UI/UX topics
- Non-database system administration

Respond with ONLY a JSON object:
{{"needs_sql": true/false, "confidence": 0.0-1.0, "query_type": "select/insert/update/analytical/other"}}
"""
            
            llm_client = UnifiedLLMClient()
            response = await llm_client.complete(
                prompt=detection_prompt.format(query=query),
                model="gpt-4o-mini",
                temperature=0.1,
                max_tokens=100
            )
            
            import json
            content = response.content.strip()
            
            # Clean JSON response
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()
            
            # Find JSON object
            if not content.startswith('{'):
                json_start = content.find('{')
                if json_start != -1:
                    content = content[json_start:]
            
            # Find matching closing brace
            if content.startswith('{'):
                brace_count = 0
                end_pos = 0
                for i, char in enumerate(content):
                    if char == '{':
                        brace_count += 1
                    elif char == '}':
                        brace_count -= 1
                        if brace_count == 0:
                            end_pos = i + 1
                            break
                if end_pos > 0:
                    content = content[:end_pos]
            
            result = json.loads(content)
            
            logger.info(f"SQL detection: {result['needs_sql']} (confidence: {result['confidence']:.2f})")
            return result['needs_sql'], result['confidence']
            
        except Exception as e:
            logger.warning(f"Intelligent SQL detection failed, falling back: {e}")
            return self.detect_sql_context(query), 0.5
    
    def format_sql_uppercase(self, sql: str) -> str:
        """Convert SQL keywords to uppercase while preserving other text"""
        # Split SQL into tokens
        # This is a simplified approach - a full SQL parser would be better
        
        # Protect strings first
        strings = []
        string_pattern = r"'[^']*'"
        
        def replace_string(match):
            strings.append(match.group(0))
            return f"__STRING_{len(strings)-1}__"
        
        sql_protected = re.sub(string_pattern, replace_string, sql)
        
        # Convert keywords to uppercase
        words = sql_protected.split()
        formatted_words = []
        
        for word in words:
            # Remove trailing punctuation for checking
            word_clean = word.rstrip(',;()').lstrip('(')
            
            if word_clean.upper() in self.uppercase_keywords:
                # Replace the clean part with uppercase
                formatted_word = word.replace(word_clean, word_clean.upper())
                formatted_words.append(formatted_word)
            else:
                formatted_words.append(word)
        
        formatted_sql = ' '.join(formatted_words)
        
        # Restore strings
        for i, string in enumerate(strings):
            formatted_sql = formatted_sql.replace(f"__STRING_{i}__", string)
        
        return formatted_sql
    
    def analyze_query(self, sql: str) -> List[SQLOptimizationSuggestion]:
        """Analyze SQL query and provide optimization suggestions"""
        suggestions = []
        sql_upper = sql.upper()
        
        # Check for anti-patterns
        anti_patterns = self.patterns.get('anti_patterns', {})
        
        # SELECT * check
        if 'SELECT *' in sql_upper or 'SELECT\n*' in sql_upper:
            suggestions.append(SQLOptimizationSuggestion(
                issue="SELECT * detected",
                suggestion="Explicitly list needed columns to reduce data scanned",
                severity="critical",
                example="SELECT user_id, amount, created_at FROM transactions"
            ))
        
        # Missing WHERE clause in non-aggregation
        if 'FROM' in sql_upper and 'WHERE' not in sql_upper and 'GROUP BY' not in sql_upper:
            if 'JOIN' not in sql_upper:  # Simple SELECT without filter
                suggestions.append(SQLOptimizationSuggestion(
                    issue="No WHERE clause detected",
                    suggestion="Add filters to reduce data scanned",
                    severity="warning",
                    example="WHERE created_at >= DATEADD('day', -30, CURRENT_DATE())"
                ))
        
        # Function on filtered column
        function_pattern = r'WHERE\s+[A-Z_]+\([a-zA-Z_]+\)\s*='
        if re.search(function_pattern, sql_upper):
            suggestions.append(SQLOptimizationSuggestion(
                issue="Function on column in WHERE clause",
                suggestion="Avoid functions on columns to enable partition pruning",
                severity="critical",
                example="Instead of WHERE DATE(created_at) = '2024-01-01', use WHERE created_at = '2024-01-01'"
            ))
        
        # Cross join detection
        if re.search(r'FROM\s+\w+\s*,\s*\w+', sql):
            suggestions.append(SQLOptimizationSuggestion(
                issue="Implicit cross join detected",
                suggestion="Use explicit JOIN with ON clause",
                severity="critical",
                example="FROM table1 t1 INNER JOIN table2 t2 ON t1.id = t2.id"
            ))
        
        # OR in JOIN condition
        if re.search(r'JOIN.*ON.*\sOR\s', sql_upper):
            suggestions.append(SQLOptimizationSuggestion(
                issue="OR condition in JOIN",
                suggestion="Split into UNION or redesign join logic",
                severity="warning"
            ))
        
        # ORDER BY without LIMIT
        if 'ORDER BY' in sql_upper and 'LIMIT' not in sql_upper:
            suggestions.append(SQLOptimizationSuggestion(
                issue="ORDER BY without LIMIT",
                suggestion="Add LIMIT or remove ORDER BY if not needed",
                severity="warning",
                example="ORDER BY created_at DESC LIMIT 1000"
            ))
        
        # DISTINCT on all columns
        if 'SELECT DISTINCT *' in sql_upper:
            suggestions.append(SQLOptimizationSuggestion(
                issue="DISTINCT * is expensive",
                suggestion="Use specific columns or GROUP BY",
                severity="warning"
            ))
        
        # Not using uppercase keywords
        lowercase_keywords = []
        for keyword in ['select', 'from', 'where', 'join', 'group by']:
            if keyword in sql.lower() and keyword.upper() not in sql:
                lowercase_keywords.append(keyword)
        
        if lowercase_keywords:
            suggestions.append(SQLOptimizationSuggestion(
                issue=f"Lowercase SQL keywords: {', '.join(lowercase_keywords)}",
                suggestion="Use UPPERCASE for all SQL keywords",
                severity="info"
            ))
        
        return suggestions
    
    def generate_optimized_query(self, request: SQLQueryRequest) -> str:
        """Generate an optimized SQL query based on request"""
        patterns = self.patterns.get('query_patterns', {})
        
        if request.query_type == QueryType.SELECT:
            if request.aggregations:
                template = patterns.get('aggregation', {}).get('template', '')
            elif request.joins:
                template = patterns.get('join_query', {}).get('template', '')
            else:
                template = patterns.get('basic_select', {}).get('template', '')
            
            # Build the query
            query_parts = []
            
            # SELECT clause
            if request.columns:
                columns = ',\n    '.join(request.columns)
            else:
                columns = "*"  # Should be avoided in production
            
            query_parts.append(f"SELECT \n    {columns}")
            
            # FROM clause
            if request.tables:
                query_parts.append(f"FROM {request.tables[0]} AS t")
            
            # JOIN clauses
            if request.joins:
                for join in request.joins:
                    join_type = join.get('type', 'INNER').upper()
                    query_parts.append(
                        f"{join_type} JOIN {join['table']} AS {join.get('alias', 'j')}\n"
                        f"    ON {join['condition']}"
                    )
            
            # WHERE clause
            if request.conditions:
                conditions = '\n    AND '.join(request.conditions)
                query_parts.append(f"WHERE {conditions}")
            
            # GROUP BY clause
            if request.aggregations:
                group_cols = [col for col in request.columns if 'SUM(' not in col and 'COUNT(' not in col]
                if group_cols:
                    query_parts.append(f"GROUP BY \n    {','.join(group_cols)}")
            
            # ORDER BY clause
            if request.order_by:
                query_parts.append(f"ORDER BY {', '.join(request.order_by)}")
            
            # LIMIT clause
            if request.limit:
                query_parts.append(f"LIMIT {request.limit}")
            
            return '\n'.join(query_parts)
        
        return "-- Query generation for this type not yet implemented"
    
    def get_optimization_tips(self, query_context: str) -> List[str]:
        """Get relevant optimization tips based on context"""
        tips = []
        context_lower = query_context.lower()
        
        if 'join' in context_lower:
            tips.extend([
                "Join on clustered/indexed columns for better performance",
                "Filter data before joining to reduce shuffle",
                "Avoid functions in JOIN conditions",
                "Use appropriate join type (INNER is usually fastest)"
            ])
        
        if 'aggregate' in context_lower or 'group' in context_lower:
            tips.extend([
                "Use window functions instead of self-joins for running totals",
                "Filter before aggregating to reduce computation",
                "Consider materialized views for frequently used aggregations",
                "Group by clustered columns when possible"
            ])
        
        if 'large' in context_lower or 'performance' in context_lower:
            tips.extend([
                "Use appropriate warehouse size for query complexity",
                "Enable result caching for repeated queries",
                "Leverage clustering keys aligned with common filters",
                "Consider partitioning large tables by date",
                "Use SAMPLE clause for development/testing"
            ])
        
        if 'date' in context_lower or 'time' in context_lower:
            tips.extend([
                "Use DATE_TRUNC for date comparisons instead of functions",
                "Filter on date ranges to enable partition pruning",
                "Avoid date functions on columns in WHERE clause"
            ])
        
        return tips
    
    def enhance_query_with_sql_context(self, query: str) -> str:
        """Enhance a query with relevant SQL optimization context"""
        if not self.detect_sql_context(query):
            return query
        
        context_parts = []
        
        # Add header
        context_parts.append("[Snowflake SQL Optimization Knowledge]")
        context_parts.append("=" * 50)
        
        # Core principles
        context_parts.append("\n⚡ Performance Best Practices:")
        context_parts.append("• SELECT only needed columns (never SELECT * in production)")
        context_parts.append("• Filter early with WHERE clauses for partition pruning")
        context_parts.append("• Join on indexed/clustered columns")
        context_parts.append("• Use UPPERCASE for all SQL keywords")
        context_parts.append("• Avoid functions on filtered columns")
        
        # Get specific tips based on query
        tips = self.get_optimization_tips(query)
        if tips:
            context_parts.append("\n🎯 Relevant Optimization Tips:")
            for tip in tips[:5]:  # Limit to 5 most relevant
                context_parts.append(f"• {tip}")
        
        # Add formatting rules
        context_parts.append("\n📝 SQL Formatting Standards:")
        context_parts.append("• One clause per line")
        context_parts.append("• Indent with 4 spaces")
        context_parts.append("• Comma at end of line")
        context_parts.append("• Table aliases for clarity")
        context_parts.append("• Meaningful column aliases")
        
        # Add common patterns if relevant
        if 'window' in query.lower() or 'rank' in query.lower():
            context_parts.append("\n🪟 Window Function Pattern:")
            context_parts.append("""ROW_NUMBER() OVER (
    PARTITION BY group_column
    ORDER BY sort_column DESC
) AS rank""")
        
        if 'cte' in query.lower() or 'with' in query.lower():
            context_parts.append("\n🔄 CTE Pattern:")
            context_parts.append("""WITH cte_name AS (
    SELECT columns
    FROM table
    WHERE conditions
)
SELECT * FROM cte_name""")
        
        # Snowflake specifics
        context_parts.append("\n❄️ Snowflake Specific:")
        context_parts.append("• Use CLUSTER BY for frequently filtered columns")
        context_parts.append("• TRANSIENT tables for staging (no time travel)")
        context_parts.append("• Result caching with identical queries")
        context_parts.append("• Appropriate warehouse sizing for query complexity")
        
        # Add anti-patterns to avoid
        context_parts.append("\n⛔ Avoid These Anti-Patterns:")
        context_parts.append("• SELECT * (explicitly list columns)")
        context_parts.append("• Functions on WHERE columns (prevents pruning)")
        context_parts.append("• OR in JOIN conditions (prevents optimization)")
        context_parts.append("• Missing WHERE clause (full table scan)")
        context_parts.append("• Unnecessary ORDER BY (expensive operation)")
        
        if context_parts:
            context = "\n".join(context_parts)
            enhanced = f"{query}\n\n{context}"
            logger.info(f"Enhanced SQL query context from {len(query)} to {len(enhanced)} chars")
            return enhanced
        
        return query
    
    def validate_sql_syntax(self, sql: str) -> Tuple[bool, List[str]]:
        """Basic SQL syntax validation"""
        issues = []
        
        # Check for basic structure
        sql_upper = sql.upper()
        
        if 'SELECT' in sql_upper:
            if 'FROM' not in sql_upper:
                issues.append("SELECT statement missing FROM clause")
        
        # Check for unmatched parentheses
        if sql.count('(') != sql.count(')'):
            issues.append("Unmatched parentheses")
        
        # Check for unmatched quotes
        if sql.count("'") % 2 != 0:
            issues.append("Unmatched single quotes")
        
        # Check JOIN has ON clause
        if 'JOIN' in sql_upper:
            join_pattern = r'JOIN\s+\w+\s+(?:AS\s+\w+\s+)?(?!ON)'
            if re.search(join_pattern, sql_upper):
                issues.append("JOIN missing ON clause")
        
        return len(issues) == 0, issues
    
    def get_query_pattern(self, pattern_name: str) -> Optional[str]:
        """Retrieve a specific query pattern template"""
        patterns = self.patterns.get('query_patterns', {})
        pattern = patterns.get(pattern_name, {})
        return pattern.get('template')
    
    def get_transformation(self, transformation_type: str, operation: str) -> Optional[str]:
        """Get a specific transformation pattern"""
        transformations = self.patterns.get('transformations', {})
        transform_group = transformations.get(transformation_type, {})
        return transform_group.get(operation)


class SQLMemoryIntegration:
    """Store and learn from successful SQL patterns"""
    
    def __init__(self, project_id: str):
        self.project_id = project_id
    
    async def store_successful_query(
        self,
        query_sql: str,
        execution_time: float,
        rows_returned: int,
        bytes_scanned: int
    ):
        """Store a successful SQL query pattern"""
        from services.memory_service import ProjectMemoryService
        
        memory_service = ProjectMemoryService(self.project_id, "sql_patterns")
        
        # Calculate efficiency score
        efficiency_score = (rows_returned / max(bytes_scanned / 1024, 1)) / max(execution_time, 0.1)
        
        await memory_service.store_memory(
            content=f"SQL Query:\n{query_sql}",
            memory_type="sql_pattern",
            metadata={
                "execution_time": execution_time,
                "rows_returned": rows_returned,
                "bytes_scanned": bytes_scanned,
                "efficiency_score": efficiency_score
            }
        )
    
    async def get_similar_queries(self, query_description: str) -> List[Dict[str, Any]]:
        """Retrieve similar successful SQL queries"""
        from services.memory_service import ProjectMemoryService
        
        memory_service = ProjectMemoryService(self.project_id, "sql_patterns")
        
        memories = await memory_service.search_memories(
            query=query_description,
            memory_types=["sql_pattern"],
            limit=5
        )
        
        return [
            {
                "query": mem.content,
                "efficiency_score": mem.metadata.get('efficiency_score', 0),
                "execution_time": mem.metadata.get('execution_time', 0)
            }
            for mem in memories
        ]