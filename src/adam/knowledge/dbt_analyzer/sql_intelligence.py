"""
SQL Intelligence Engine
Understands SQL transformations and generates intelligent descriptions
Uses sqlglot for proper SQL parsing and Claude for natural language generation
"""

import logging
import re
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum

try:
    import sqlglot
    from sqlglot import exp, parse_one
    SQLGLOT_AVAILABLE = True
except ImportError:
    SQLGLOT_AVAILABLE = False
    print("Warning: sqlglot not installed. Run: pip install sqlglot")

from adam.llm.client import UnifiedLLMClient

logger = logging.getLogger(__name__)

class TransformationType(Enum):
    """Types of SQL transformations"""
    CASE_WHEN = "case_when"
    WINDOW_FUNCTION = "window_function"
    AGGREGATION = "aggregation"
    CAST = "cast"
    COALESCE = "coalesce"
    CONCAT = "concat"
    DATE_FUNCTION = "date_function"
    ARITHMETIC = "arithmetic"
    SUBSTRING = "substring"
    CONDITIONAL = "conditional"
    SIMPLE_SELECT = "simple_select"

@dataclass
class ColumnTransformation:
    """Represents a column's transformation logic"""
    column_name: str
    transformation_type: TransformationType
    source_columns: List[str]
    logic_description: str
    sql_snippet: str
    complexity_score: int
    business_logic: Optional[str] = None

class SQLIntelligenceEngine:
    """
    Analyzes SQL to understand column transformations and generate intelligent descriptions
    """

    def __init__(self):
        self.llm_client = UnifiedLLMClient()

    async def analyze_column_sql(self, sql: str, column_name: str, model_name: Optional[str] = None) -> ColumnTransformation:
        """
        Analyze a SQL query to understand how a specific column is derived
        """
        if not SQLGLOT_AVAILABLE:
            logger.warning("sqlglot not available, using fallback analysis")
            return await self._fallback_analyze(sql, column_name)

        try:
            # Parse SQL with sqlglot
            parsed = parse_one(sql, read="duckdb")  # DuckDB dialect is permissive

            # Find the column in SELECT clause
            transformation = self._extract_column_transformation(parsed, column_name)

            if not transformation:
                return await self._fallback_analyze(sql, column_name)

            # Generate intelligent description using Claude
            transformation.business_logic = await self._generate_business_description(
                transformation, model_name
            )

            return transformation

        except Exception as e:
            logger.warning(f"Error parsing SQL with sqlglot: {e}")
            return await self._fallback_analyze(sql, column_name)

    def _extract_column_transformation(self, parsed: exp.Expression, column_name: str) -> Optional[ColumnTransformation]:
        """
        Extract transformation logic for a specific column from parsed SQL
        """
        try:
            # Find SELECT clause
            select_expressions = []

            for node in parsed.find_all(exp.Select):
                for projection in node.expressions:
                    # Check if this is our column (either by alias or name)
                    alias = projection.alias if hasattr(projection, 'alias') else None

                    if alias == column_name or str(projection).startswith(column_name):
                        # Found our column! Analyze it
                        return self._analyze_expression(projection, column_name)

            return None

        except Exception as e:
            logger.error(f"Error extracting transformation: {e}")
            return None

    def _analyze_expression(self, expr: exp.Expression, column_name: str) -> ColumnTransformation:
        """
        Analyze a specific SQL expression to understand the transformation
        """
        sql_snippet = str(expr)
        source_columns = self._extract_source_columns(expr)

        # Detect transformation type
        transformation_type = self._detect_transformation_type(expr)

        # Extract logic description
        logic_description = self._extract_logic_description(expr, transformation_type)

        # Calculate complexity
        complexity_score = self._calculate_complexity(expr)

        return ColumnTransformation(
            column_name=column_name,
            transformation_type=transformation_type,
            source_columns=source_columns,
            logic_description=logic_description,
            sql_snippet=sql_snippet,
            complexity_score=complexity_score
        )

    def _extract_source_columns(self, expr: exp.Expression) -> List[str]:
        """Extract all source columns referenced in an expression"""
        columns = []

        # Find all column references
        for col in expr.find_all(exp.Column):
            col_name = col.name
            if col_name and col_name not in columns:
                columns.append(col_name)

        return columns

    def _detect_transformation_type(self, expr: exp.Expression) -> TransformationType:
        """Detect what type of transformation this is"""

        # Check for CASE WHEN
        if expr.find(exp.Case):
            return TransformationType.CASE_WHEN

        # Check for window functions
        if expr.find(exp.Window):
            return TransformationType.WINDOW_FUNCTION

        # Check for aggregations
        if any(expr.find(agg) for agg in [exp.Sum, exp.Avg, exp.Count, exp.Max, exp.Min]):
            return TransformationType.AGGREGATION

        # Check for CAST
        if expr.find(exp.Cast):
            return TransformationType.CAST

        # Check for COALESCE
        if expr.find(exp.Coalesce):
            return TransformationType.COALESCE

        # Check for CONCAT
        if expr.find(exp.Concat):
            return TransformationType.CONCAT

        # Check for date functions
        date_funcs = [exp.DateAdd, exp.DateDiff, exp.CurrentDate, exp.CurrentTimestamp]
        if any(expr.find(func) for func in date_funcs):
            return TransformationType.DATE_FUNCTION

        # Check for arithmetic
        if any(expr.find(op) for op in [exp.Add, exp.Sub, exp.Mul, exp.Div]):
            return TransformationType.ARITHMETIC

        # Check for SUBSTRING
        if expr.find(exp.Substring):
            return TransformationType.SUBSTRING

        # Default to simple select
        return TransformationType.SIMPLE_SELECT

    def _extract_logic_description(self, expr: exp.Expression, trans_type: TransformationType) -> str:
        """Extract a human-readable description of the logic"""

        if trans_type == TransformationType.CASE_WHEN:
            return self._describe_case_when(expr)

        elif trans_type == TransformationType.WINDOW_FUNCTION:
            return self._describe_window_function(expr)

        elif trans_type == TransformationType.AGGREGATION:
            return self._describe_aggregation(expr)

        elif trans_type == TransformationType.CAST:
            cast_expr = expr.find(exp.Cast)
            return f"Cast to {cast_expr.to.name if cast_expr else 'unknown'}"

        elif trans_type == TransformationType.COALESCE:
            coalesce = expr.find(exp.Coalesce)
            num_args = len(coalesce.expressions) if coalesce else 0
            return f"Coalesce with {num_args} fallback values"

        elif trans_type == TransformationType.ARITHMETIC:
            return "Arithmetic calculation"

        else:
            return "Direct column reference or simple transformation"

    def _describe_case_when(self, expr: exp.Expression) -> str:
        """Describe CASE WHEN logic"""
        case_expr = expr.find(exp.Case)
        if not case_expr:
            return "Conditional logic"

        num_conditions = len([c for c in case_expr.find_all(exp.If)])
        has_else = case_expr.default is not None

        return f"Conditional with {num_conditions} conditions{' and ELSE clause' if has_else else ''}"

    def _describe_window_function(self, expr: exp.Expression) -> str:
        """Describe window function logic"""
        window = expr.find(exp.Window)
        if not window:
            return "Window function"

        # Get function name
        func_name = "Unknown"
        if window.this:
            func_name = window.this.key if hasattr(window.this, 'key') else str(window.this)

        # Get partition and order
        partition_cols = []
        order_cols = []

        if hasattr(window, 'partition_by') and window.partition_by:
            partition_cols = [str(col) for col in window.partition_by]

        if hasattr(window, 'order') and window.order:
            order_cols = [str(col) for col in window.order.expressions]

        desc = f"{func_name} window function"
        if partition_cols:
            desc += f" partitioned by {', '.join(partition_cols)}"
        if order_cols:
            desc += f" ordered by {', '.join(order_cols)}"

        return desc

    def _describe_aggregation(self, expr: exp.Expression) -> str:
        """Describe aggregation logic"""
        agg_types = {
            exp.Sum: "Sum",
            exp.Avg: "Average",
            exp.Count: "Count",
            exp.Max: "Maximum",
            exp.Min: "Minimum"
        }

        for agg_class, agg_name in agg_types.items():
            if expr.find(agg_class):
                return f"{agg_name} aggregation"

        return "Aggregation"

    def _calculate_complexity(self, expr: exp.Expression) -> int:
        """Calculate complexity score for an expression"""
        score = 0

        # Base complexity
        score += 1

        # CASE WHEN complexity
        case_expr = expr.find(exp.Case)
        if case_expr:
            num_conditions = len([c for c in case_expr.find_all(exp.If)])
            score += num_conditions * 2

        # Window function complexity
        if expr.find(exp.Window):
            score += 3

        # Nested subqueries
        score += len(list(expr.find_all(exp.Subquery))) * 3

        # Number of functions
        score += len(list(expr.find_all(exp.Func)))

        # Number of source columns
        score += len(self._extract_source_columns(expr))

        return score

    async def _generate_business_description(self, transformation: ColumnTransformation,
                                            model_name: Optional[str]) -> str:
        """
        Use Claude to generate a business-friendly description of the transformation
        """
        try:
            context_info = f" in model '{model_name}'" if model_name else ""

            prompt = f"""Generate a clear, business-friendly description for this database column{context_info}.

Column: {transformation.column_name}
Transformation Type: {transformation.transformation_type.value}
Logic: {transformation.logic_description}
Source Columns: {', '.join(transformation.source_columns) if transformation.source_columns else 'None'}

SQL:
```sql
{transformation.sql_snippet}
```

Generate a concise description (under 150 characters) that explains:
1. WHAT the column represents from a business perspective
2. HOW it's derived (if relevant)

Be specific and avoid technical jargon. Focus on the business meaning.
Respond with ONLY the description, no other text."""

            response = await self.llm_client.complete(
                prompt=prompt,
                model="claude-sonnet-4-5-20250929",  # Use Sonnet for code understanding
                temperature=0.3,
                max_tokens=100
            )

            return response.content.strip()

        except Exception as e:
            logger.error(f"Error generating business description: {e}")
            return transformation.logic_description

    async def _fallback_analyze(self, sql: str, column_name: str) -> ColumnTransformation:
        """Fallback analysis when sqlglot is not available"""

        # Extract the column definition with regex
        pattern = rf'{column_name}\s+as\s+(.+?)(?:,|\s+from|\s+$)'
        match = re.search(pattern, sql, re.IGNORECASE | re.DOTALL)

        if match:
            sql_snippet = match.group(1).strip()
        else:
            sql_snippet = f"Unknown transformation for {column_name}"

        # Basic pattern detection
        if 'case' in sql_snippet.lower():
            trans_type = TransformationType.CASE_WHEN
            logic = "Conditional logic"
        elif any(func in sql_snippet.lower() for func in ['row_number', 'rank', 'lag', 'lead']):
            trans_type = TransformationType.WINDOW_FUNCTION
            logic = "Window function"
        elif any(func in sql_snippet.lower() for func in ['sum', 'avg', 'count', 'max', 'min']):
            trans_type = TransformationType.AGGREGATION
            logic = "Aggregation"
        else:
            trans_type = TransformationType.SIMPLE_SELECT
            logic = "Direct reference"

        # Extract source columns (simple regex)
        source_cols = re.findall(r'\b([a-z_][a-z0-9_]*)\b', sql_snippet.lower())
        source_cols = [c for c in source_cols if c not in ['as', 'and', 'or', 'when', 'then', 'else', 'case', 'end']]

        return ColumnTransformation(
            column_name=column_name,
            transformation_type=trans_type,
            source_columns=list(set(source_cols)),
            logic_description=logic,
            sql_snippet=sql_snippet,
            complexity_score=len(source_cols) + (3 if trans_type != TransformationType.SIMPLE_SELECT else 0)
        )

    async def batch_analyze_model(self, sql: str, model_name: str) -> Dict[str, ColumnTransformation]:
        """
        Analyze all columns in a model's SQL
        """
        transformations = {}

        try:
            if SQLGLOT_AVAILABLE:
                parsed = parse_one(sql, read="duckdb")

                # Find all SELECT expressions
                for node in parsed.find_all(exp.Select):
                    for projection in node.expressions:
                        # Get column name (alias or expression)
                        alias = projection.alias if hasattr(projection, 'alias') else None

                        if alias:
                            column_name = alias
                        else:
                            # Try to extract column name from expression
                            column_name = str(projection).split()[0]

                        if column_name:
                            transformation = await self.analyze_column_sql(sql, column_name, model_name)
                            transformations[column_name] = transformation

            else:
                # Fallback: extract columns from SELECT clause with regex
                select_pattern = r'SELECT\s+(.*?)\s+FROM'
                match = re.search(select_pattern, sql, re.IGNORECASE | re.DOTALL)

                if match:
                    select_clause = match.group(1)
                    # Split by comma (simple approach)
                    for col_def in select_clause.split(','):
                        # Extract alias if present
                        as_match = re.search(r'as\s+([a-z_][a-z0-9_]*)', col_def, re.IGNORECASE)
                        if as_match:
                            column_name = as_match.group(1)
                            transformation = await self._fallback_analyze(sql, column_name)
                            transformations[column_name] = transformation

        except Exception as e:
            logger.error(f"Error in batch analysis: {e}")

        return transformations