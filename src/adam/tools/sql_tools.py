"""
SQL Analysis Tools for ADAM
Helps analytics engineers optimize queries, format SQL, and identify issues
"""
import re
import sqlparse
from typing import List, Dict, Optional, Tuple, Set
from dataclasses import dataclass
from enum import Enum
import asyncio
from pathlib import Path
import sys

# Add parent path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from adam.llm.client import UnifiedLLMClient


class IssueLevel(Enum):
    """Severity levels for SQL issues"""
    ERROR = "error"
    WARNING = "warning"
    SUGGESTION = "suggestion"
    INFO = "info"


@dataclass
class SQLIssue:
    """Represents an issue found in SQL query"""
    level: IssueLevel
    message: str
    line_number: Optional[int] = None
    suggestion: Optional[str] = None
    estimated_impact: Optional[str] = None


@dataclass
class QueryMetrics:
    """Metrics about a SQL query"""
    line_count: int
    cte_count: int
    join_count: int
    subquery_count: int
    distinct_count: int
    complexity_score: int  # 1-10 scale


class SQLAnalyzer:
    """
    Analyzes SQL queries for performance issues, anti-patterns, and optimization opportunities
    Specifically tuned for Snowflake, BigQuery, and Redshift
    """
    
    def __init__(self, dialect: str = "snowflake"):
        self.dialect = dialect.lower()
        self.llm_client = None  # Lazy load when needed
        
    async def _get_llm_client(self):
        """Lazy load LLM client"""
        if not self.llm_client:
            self.llm_client = UnifiedLLMClient()
        return self.llm_client
        
    def analyze_query(self, query: str, context: Optional[Dict] = None) -> Tuple[List[SQLIssue], QueryMetrics]:
        """
        Analyze SQL query for issues and gather metrics
        
        Args:
            query: SQL query to analyze
            context: Optional context (table sizes, indexes, etc.)
            
        Returns:
            Tuple of (issues list, query metrics)
        """
        issues = []
        
        # Clean and parse query
        query_upper = query.upper()
        parsed = sqlparse.parse(query)[0] if sqlparse.parse(query) else None
        
        # Gather metrics
        metrics = self._calculate_metrics(query, parsed)
        
        # Check for various issues
        issues.extend(self._check_select_star(query_upper, query))
        issues.extend(self._check_missing_where(query_upper, parsed))
        issues.extend(self._check_implicit_cross_joins(query_upper))
        issues.extend(self._check_expensive_operations(query_upper))
        issues.extend(self._check_subquery_issues(query_upper, query))
        issues.extend(self._check_distinct_usage(query_upper))
        issues.extend(self._check_cte_usage(query_upper, query))
        
        # Dialect-specific checks
        if self.dialect == "snowflake":
            issues.extend(self._check_snowflake_specific(query_upper, query))
        elif self.dialect == "bigquery":
            issues.extend(self._check_bigquery_specific(query_upper, query))
            
        # Add complexity warning if needed
        if metrics.complexity_score >= 8:
            issues.append(SQLIssue(
                level=IssueLevel.WARNING,
                message="Query is highly complex. Consider breaking into smaller queries or views",
                suggestion="Split into multiple CTEs or create intermediate views"
            ))
            
        return issues, metrics
    
    def _calculate_metrics(self, query: str, parsed) -> QueryMetrics:
        """Calculate query complexity metrics"""
        query_upper = query.upper()
        
        # Count various elements
        line_count = len(query.split('\n'))
        
        # Count CTEs more accurately
        cte_pattern = r'\b\w+\s+AS\s*\('
        cte_matches = re.findall(cte_pattern, query_upper)
        cte_count = len([m for m in cte_matches if 'WITH' in query_upper[:query_upper.find(m)]])
        
        # Count joins (avoid double counting)
        join_count = len(re.findall(r'\b(?:LEFT\s+|RIGHT\s+|FULL\s+|INNER\s+|CROSS\s+)?JOIN\b', query_upper))
        
        subquery_count = query.count('(SELECT')
        distinct_count = query_upper.count('DISTINCT')
        
        # Calculate complexity score (1-10)
        complexity_score = min(10, max(1, (
            (line_count // 30) +  # Lower threshold for line count
            (cte_count) +         # Each CTE adds complexity
            (join_count) +        # Each join adds complexity
            (subquery_count) +    # Each subquery adds complexity
            (distinct_count // 2) # DISTINCT operations add some complexity
        )))
        
        return QueryMetrics(
            line_count=line_count,
            cte_count=cte_count,
            join_count=join_count,
            subquery_count=subquery_count,
            distinct_count=distinct_count,
            complexity_score=max(1, complexity_score)
        )
    
    def _check_select_star(self, query_upper: str, query: str) -> List[SQLIssue]:
        """Check for SELECT * usage"""
        issues = []
        
        # Find all SELECT * occurrences
        pattern = r'SELECT\s+\*'
        matches = list(re.finditer(pattern, query_upper))
        
        if matches:
            for match in matches:
                line_num = query[:match.start()].count('\n') + 1
                issues.append(SQLIssue(
                    level=IssueLevel.WARNING,
                    message="Avoid SELECT *, specify needed columns",
                    line_number=line_num,
                    suggestion="List specific columns to reduce data transfer and improve performance",
                    estimated_impact="Can reduce query time by 20-80% depending on table width"
                ))
                
        return issues
    
    def _check_missing_where(self, query_upper: str, parsed) -> List[SQLIssue]:
        """Check for missing WHERE clause in DELETE/UPDATE"""
        issues = []
        
        if 'DELETE' in query_upper and 'WHERE' not in query_upper:
            issues.append(SQLIssue(
                level=IssueLevel.ERROR,
                message="DELETE without WHERE clause will delete all rows!",
                suggestion="Add WHERE clause or use TRUNCATE if you really want to delete all rows",
                estimated_impact="Potential data loss"
            ))
            
        if 'UPDATE' in query_upper and 'WHERE' not in query_upper:
            issues.append(SQLIssue(
                level=IssueLevel.ERROR,
                message="UPDATE without WHERE clause will update all rows!",
                suggestion="Add WHERE clause to target specific rows",
                estimated_impact="Potential data corruption"
            ))
            
        return issues
    
    def _check_implicit_cross_joins(self, query_upper: str) -> List[SQLIssue]:
        """Check for implicit cross joins (comma-separated tables)"""
        issues = []
        
        # Look for FROM table1, table2 pattern
        pattern = r'FROM\s+\w+\s*,\s*\w+'
        if re.search(pattern, query_upper):
            issues.append(SQLIssue(
                level=IssueLevel.WARNING,
                message="Implicit cross join detected (comma-separated tables)",
                suggestion="Use explicit JOIN syntax for clarity and to avoid cartesian products",
                estimated_impact="Can cause massive performance issues with large tables"
            ))
            
        return issues
    
    def _check_expensive_operations(self, query_upper: str) -> List[SQLIssue]:
        """Check for known expensive operations"""
        issues = []
        
        # Check for LIKE with leading wildcard
        if re.search(r"LIKE\s+'%[^']", query_upper):
            issues.append(SQLIssue(
                level=IssueLevel.WARNING,
                message="LIKE with leading wildcard prevents index usage",
                suggestion="Consider full-text search or redesigning the query",
                estimated_impact="Forces full table scan"
            ))
            
        # Check for NOT IN with subquery
        if 'NOT IN' in query_upper and '(SELECT' in query_upper:
            issues.append(SQLIssue(
                level=IssueLevel.WARNING,
                message="NOT IN with subquery can be slow and has NULL handling issues",
                suggestion="Use NOT EXISTS or LEFT JOIN with NULL check instead",
                estimated_impact="Can be 10x slower than NOT EXISTS"
            ))
            
        # Check for OR in JOIN conditions
        if re.search(r'JOIN.*?\bON\b.*?\bOR\b', query_upper, re.DOTALL):
            issues.append(SQLIssue(
                level=IssueLevel.WARNING,
                message="OR in JOIN condition prevents efficient join algorithms",
                suggestion="Consider UNION or redesigning the join logic",
                estimated_impact="Forces nested loop join instead of hash/merge join"
            ))
            
        return issues
    
    def _check_subquery_issues(self, query_upper: str, query: str) -> List[SQLIssue]:
        """Check for subquery anti-patterns"""
        issues = []
        
        # Count subqueries in SELECT clause
        # Find the main FROM (last one that's not inside parentheses)
        from_positions = []
        paren_depth = 0
        i = 0
        while i < len(query_upper):
            if query_upper[i] == '(':
                paren_depth += 1
            elif query_upper[i] == ')':
                paren_depth -= 1
            elif query_upper[i:i+4] == 'FROM' and paren_depth == 0:
                # Found a FROM at depth 0 (not inside subquery)
                from_positions.append(i)
            i += 1
            
        if from_positions:
            # Get content before the last main FROM
            main_from_pos = from_positions[-1]
            select_clause = query_upper[:main_from_pos]
            select_subqueries = select_clause.count('(SELECT')
            if select_subqueries > 2:
                issues.append(SQLIssue(
                    level=IssueLevel.WARNING,
                    message=f"Multiple subqueries ({select_subqueries}) in SELECT clause",
                    suggestion="Consider using JOINs or CTEs instead",
                    estimated_impact="Each subquery executes once per row"
                ))
            
        # Check for correlated subqueries
        if re.search(r'\(SELECT.*?WHERE.*?=.*?[a-z]+\.[a-z]+.*?\)', query, re.IGNORECASE):
            issues.append(SQLIssue(
                level=IssueLevel.WARNING,
                message="Possible correlated subquery detected",
                suggestion="Consider rewriting as JOIN or using window functions",
                estimated_impact="Executes once per row, can be very slow"
            ))
            
        return issues
    
    def _check_distinct_usage(self, query_upper: str) -> List[SQLIssue]:
        """Check for DISTINCT usage patterns"""
        issues = []
        
        distinct_count = query_upper.count('DISTINCT')
        if distinct_count > 2:
            issues.append(SQLIssue(
                level=IssueLevel.WARNING,
                message=f"Multiple DISTINCT operations ({distinct_count}) detected",
                suggestion="Consider if all DISTINCTs are necessary, or use GROUP BY",
                estimated_impact="Each DISTINCT requires sorting, multiple can be very expensive"
            ))
            
        # Check for DISTINCT with many columns
        if re.search(r'SELECT\s+DISTINCT\s+.{100,}FROM', query_upper, re.DOTALL):
            issues.append(SQLIssue(
                level=IssueLevel.WARNING,
                message="DISTINCT on many columns detected",
                suggestion="Consider if you need all columns or can use GROUP BY on key columns",
                estimated_impact="Expensive sort operation on wide rows"
            ))
            
        return issues
    
    def _check_cte_usage(self, query_upper: str, query: str) -> List[SQLIssue]:
        """Check CTE usage patterns"""
        issues = []
        
        # Find CTE definitions - handle both WITH and comma-separated CTEs
        cte_names = []
        
        # Find first CTE after WITH
        with_match = re.search(r'WITH\s+(\w+)\s+AS', query_upper)
        if with_match:
            cte_names.append(with_match.group(1))
            
        # Find comma-separated CTEs
        comma_cte_pattern = r',\s*(\w+)\s+AS\s*\('
        cte_names.extend(re.findall(comma_cte_pattern, query_upper))
        
        # Check if CTEs are used
        for cte in cte_names:
            # Count usage in main query (after all CTE definitions)
            # Find the end of CTE definitions (before main SELECT)
            cte_end_pattern = r'\)\s*(?:,|\s+SELECT)'
            last_cte_match = None
            for match in re.finditer(cte_end_pattern, query_upper):
                last_cte_match = match
                
            if last_cte_match:
                main_query = query_upper[last_cte_match.end():]
            else:
                main_query = query_upper
                
            # Count usage in main query
            usage_pattern = r'\b' + cte + r'\b'
            usage_count = len(re.findall(usage_pattern, main_query))
            
            if usage_count == 0:  # Only defined, never used
                issues.append(SQLIssue(
                    level=IssueLevel.WARNING,
                    message=f"CTE '{cte.lower()}' is defined but never used",
                    suggestion="Remove unused CTEs",
                    estimated_impact="Unnecessary computation"
                ))
            elif usage_count > 5:  # Used many times
                issues.append(SQLIssue(
                    level=IssueLevel.WARNING,
                    message=f"CTE '{cte.lower()}' is referenced {usage_count} times",
                    suggestion="Consider materializing as a temporary table or view",
                    estimated_impact="CTE may be recalculated multiple times"
                ))
                
        return issues
    
    def _check_snowflake_specific(self, query_upper: str, query: str) -> List[SQLIssue]:
        """Snowflake-specific optimizations"""
        issues = []
        
        # Check for missing clustering keys hint
        if 'WHERE' in query_upper and 'DATE' in query_upper:
            if not any(hint in query_upper for hint in ['CLUSTER BY', 'CLUSTERING']):
                issues.append(SQLIssue(
                    level=IssueLevel.INFO,
                    message="Query filters on date but no clustering key mentioned",
                    suggestion="Ensure table has clustering on date columns for better performance",
                    estimated_impact="Can improve performance by 10-100x for date range queries"
                ))
                
        # Check for FLATTEN without LATERAL
        if 'FLATTEN' in query_upper and 'LATERAL' not in query_upper:
            issues.append(SQLIssue(
                level=IssueLevel.WARNING,
                message="FLATTEN should be used with LATERAL for clarity",
                suggestion="Use LATERAL FLATTEN for better readability",
                estimated_impact="No performance impact, but improves clarity"
            ))
            
        # Check for missing RESULT_SCAN optimization
        if 'SHOW' in query_upper or 'DESC' in query_upper:
            issues.append(SQLIssue(
                level=IssueLevel.INFO,
                message="SHOW/DESC results can be queried with RESULT_SCAN()",
                suggestion="Use RESULT_SCAN(LAST_QUERY_ID()) to query metadata results",
                estimated_impact="Enables filtering and joining metadata results"
            ))
            
        return issues
    
    def _check_bigquery_specific(self, query_upper: str, query: str) -> List[SQLIssue]:
        """BigQuery-specific optimizations"""
        issues = []
        
        # Check for missing partitioning filter
        if '_TABLE_SUFFIX' not in query_upper and 'WHERE' in query_upper:
            date_patterns = ['DATE', 'TIMESTAMP', 'DATETIME']
            if any(p in query_upper for p in date_patterns):
                issues.append(SQLIssue(
                    level=IssueLevel.INFO,
                    message="Consider using table partitioning for date-based queries",
                    suggestion="Use _TABLE_SUFFIX or partitioned tables to reduce scan size",
                    estimated_impact="Can reduce costs by 90%+ for large tables"
                ))
                
        # Check for SELECT without LIMIT in development
        if 'LIMIT' not in query_upper and 'SELECT' in query_upper:
            issues.append(SQLIssue(
                level=IssueLevel.INFO,
                message="No LIMIT clause in BigQuery query",
                suggestion="Add LIMIT during development to control costs",
                estimated_impact="Prevents accidental full table scans"
            ))
            
        return issues
    
    async def suggest_optimizations(self, query: str, issues: List[SQLIssue]) -> str:
        """
        Use LLM to suggest query optimizations based on issues found
        
        Args:
            query: Original SQL query
            issues: List of issues found
            
        Returns:
            Optimized query suggestion
        """
        if not issues:
            return query
            
        client = await self._get_llm_client()
        
        # Build context from issues
        issue_summary = "\n".join([
            f"- {issue.level.value.upper()}: {issue.message}"
            for issue in issues[:5]  # Limit to top 5 issues
        ])
        
        prompt = f"""As an expert in SQL optimization, help optimize this {self.dialect} query based on these issues:

Issues found:
{issue_summary}

Original query:
```sql
{query[:1000]}  # Limit query length
```

Provide an optimized version that addresses the main performance issues. Include comments explaining the changes."""

        response = await client.complete(
            prompt=prompt,
            model="grok-4",  # Best for SQL optimization
            temperature=0.3
        )
        
        return response.content


class SQLFormatter:
    """
    Formats SQL according to team standards and best practices
    Supports different style guides (dbt, general, compact)
    """
    
    def __init__(self, style: str = "dbt"):
        self.style = style
        self.indent_width = 2 if style == "dbt" else 4
        
    def format_query(self, query: str, max_line_length: int = 80) -> str:
        """
        Format SQL query according to style guide
        
        Args:
            query: SQL query to format
            max_line_length: Maximum line length (default 80)
            
        Returns:
            Formatted SQL query
        """
        # Use sqlparse for initial formatting
        formatted = sqlparse.format(
            query,
            reindent=True,
            keyword_case='upper',
            identifier_case='lower',
            indent_width=self.indent_width,
            wrap_after=max_line_length
        )
        
        # Apply style-specific formatting
        if self.style == "dbt":
            formatted = self._apply_dbt_style(formatted)
        elif self.style == "compact":
            formatted = self._apply_compact_style(formatted)
            
        return formatted.strip()
    
    def _apply_dbt_style(self, query: str) -> str:
        """Apply dbt SQL style guide"""
        lines = query.split('\n')
        formatted_lines = []
        
        for line in lines:
            # CTEs on new lines
            line = re.sub(r',\s*(\w+)\s+AS', r',\n\1 AS', line)
            
            # Commas at the beginning of lines
            if ',' in line and 'SELECT' in line:
                # Move commas to start of next line
                parts = line.split(',')
                if len(parts) > 1:
                    formatted_lines.append(parts[0])
                    for part in parts[1:]:
                        formatted_lines.append('  ,' + part)
                    continue
                    
            formatted_lines.append(line)
            
        # Join lines and apply final formatting
        result = '\n'.join(formatted_lines)
        
        # Ensure CTEs are properly formatted
        result = re.sub(r'WITH\s+', 'WITH\n  ', result)
        result = re.sub(r'\)\s*,\s*(\w+)', r'),\n\n  \1', result)
        
        return result
    
    def _apply_compact_style(self, query: str) -> str:
        """Apply compact style (minimal whitespace)"""
        # Remove extra blank lines
        lines = [line for line in query.split('\n') if line.strip()]
        return '\n'.join(lines)
    
    def validate_syntax(self, query: str) -> List[str]:
        """
        Basic SQL syntax validation
        
        Returns:
            List of syntax errors (empty if valid)
        """
        errors = []
        
        # Check for basic syntax issues
        if not query.strip():
            errors.append("Empty query")
            
        # Check parentheses balance
        if query.count('(') != query.count(')'):
            errors.append("Unbalanced parentheses")
            
        # Check quotes balance
        single_quotes = query.count("'") % 2
        double_quotes = query.count('"') % 2
        if single_quotes != 0:
            errors.append("Unbalanced single quotes")
        if double_quotes != 0:
            errors.append("Unbalanced double quotes")
            
        # Check for common syntax errors
        query_upper = query.upper()
        if 'FORM' in query_upper and 'FROM' not in query_upper:
            errors.append("Possible typo: 'FORM' should be 'FROM'")
            
        if 'WEHRE' in query_upper:
            errors.append("Typo: 'WEHRE' should be 'WHERE'")
            
        return errors


class SQLOptimizer:
    """
    Advanced SQL optimizer that combines analysis and LLM suggestions
    """
    
    def __init__(self, dialect: str = "snowflake"):
        self.analyzer = SQLAnalyzer(dialect)
        self.formatter = SQLFormatter("dbt")
        self.dialect = dialect
        
    async def optimize_query(self, query: str, context: Optional[Dict] = None) -> Dict:
        """
        Complete query optimization pipeline
        
        Args:
            query: SQL query to optimize
            context: Optional context (table sizes, current performance, etc.)
            
        Returns:
            Dict with optimized query and recommendations
        """
        # Analyze query
        issues, metrics = self.analyzer.analyze_query(query, context)
        
        # Get LLM suggestions if there are issues
        optimized_query = query
        if issues:
            optimized_query = await self.analyzer.suggest_optimizations(query, issues)
            
        # Format the optimized query
        formatted_query = self.formatter.format_query(optimized_query)
        
        # Build recommendations
        recommendations = self._build_recommendations(issues, metrics)
        
        return {
            'original_query': query,
            'optimized_query': formatted_query,
            'issues': issues,
            'metrics': metrics,
            'recommendations': recommendations,
            'estimated_improvement': self._estimate_improvement(issues)
        }
    
    def _build_recommendations(self, issues: List[SQLIssue], metrics: QueryMetrics) -> List[str]:
        """Build actionable recommendations"""
        recommendations = []
        
        # High-priority issues
        errors = [i for i in issues if i.level == IssueLevel.ERROR]
        if errors:
            recommendations.append("🚨 Fix critical errors before running in production")
            
        # Performance recommendations
        warnings = [i for i in issues if i.level == IssueLevel.WARNING]
        if len(warnings) >= 3:
            recommendations.append("⚠️  Multiple performance issues found - consider refactoring")
        elif len(warnings) >= 1:
            recommendations.append("⚡ Address performance warnings for better query efficiency")
            
        # Complexity recommendations
        if metrics.complexity_score >= 8:
            recommendations.append("📊 Query is complex - consider breaking into views or CTEs")
        elif metrics.complexity_score >= 5:
            recommendations.append("📈 Moderately complex query - ensure good indexing")
            
        if metrics.cte_count > 5:
            recommendations.append("🔄 Many CTEs - ensure each adds value and isn't redundant")
            
        if metrics.join_count > 6:
            recommendations.append("🔗 Many joins - verify all are necessary and indexed")
        elif metrics.join_count >= 3:
            recommendations.append("🔗 Multiple joins detected - check join order for optimization")
            
        # If no specific recommendations, give general advice
        if not recommendations and issues:
            recommendations.append("🔍 Review identified issues for potential optimizations")
            
        return recommendations
    
    def _estimate_improvement(self, issues: List[SQLIssue]) -> str:
        """Estimate potential performance improvement"""
        if not issues:
            return "Query appears well-optimized"
            
        # Count issue types
        errors = sum(1 for i in issues if i.level == IssueLevel.ERROR)
        warnings = sum(1 for i in issues if i.level == IssueLevel.WARNING)
        
        if errors > 0:
            return "Critical issues must be fixed"
        elif warnings >= 3:
            return "Potential for 2-10x performance improvement"
        elif warnings >= 1:
            return "Potential for 20-50% performance improvement"
        else:
            return "Minor optimizations possible"


# Convenience function
async def analyze_sql(query: str, dialect: str = "snowflake") -> Dict:
    """Quick function to analyze any SQL query"""
    optimizer = SQLOptimizer(dialect)
    return await optimizer.optimize_query(query)