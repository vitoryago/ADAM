"""
SQL Optimizer for Redshift and Snowflake
Provides intelligent optimization suggestions for DBT models
"""

import re
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import json
import logging

logger = logging.getLogger(__name__)

@dataclass
class OptimizationSuggestion:
    """Represents an optimization suggestion"""
    severity: str  # CRITICAL, HIGH, MEDIUM, LOW
    category: str  # performance, cost, maintenance
    title: str
    description: str
    current_pattern: str
    suggested_pattern: str
    estimated_impact: str
    platform: str  # redshift, snowflake, both


class SQLOptimizer:
    """
    Intelligent SQL optimizer for Redshift and Snowflake
    Analyzes SQL patterns and provides platform-specific optimizations
    """

    def __init__(self):
        self.redshift_rules = self._load_redshift_rules()
        self.snowflake_rules = self._load_snowflake_rules()
        self.common_rules = self._load_common_rules()

    def _load_redshift_rules(self) -> List[Dict]:
        """Load Redshift-specific optimization rules"""
        return [
            {
                "name": "distribution_key_optimization",
                "pattern": r"JOIN\s+(\w+)\s+ON\s+\w+\.(\w+)\s*=\s*\w+\.(\w+)",
                "check": self._check_distribution_key,
                "suggestion": OptimizationSuggestion(
                    severity="HIGH",
                    category="performance",
                    title="Optimize Distribution Key",
                    description="Large tables should use DISTKEY on frequently joined columns",
                    current_pattern="Table without optimal DISTKEY",
                    suggested_pattern="ALTER TABLE {table} ALTER DISTKEY {column};",
                    estimated_impact="30-50% faster joins on large tables",
                    platform="redshift"
                )
            },
            {
                "name": "sort_key_optimization",
                "pattern": r"WHERE.*?(\w+)\s*(?:=|>|<|>=|<=|BETWEEN)",
                "check": self._check_sort_key,
                "suggestion": OptimizationSuggestion(
                    severity="MEDIUM",
                    category="performance",
                    title="Add Sort Key",
                    description="Add SORTKEY on frequently filtered columns",
                    current_pattern="Filtering without SORTKEY",
                    suggested_pattern="ALTER TABLE {table} ALTER COMPOUND SORTKEY ({columns});",
                    estimated_impact="20-40% faster WHERE clauses",
                    platform="redshift"
                )
            },
            {
                "name": "compression_encoding",
                "pattern": r"CREATE\s+TABLE\s+(\w+)",
                "check": self._check_compression,
                "suggestion": OptimizationSuggestion(
                    severity="LOW",
                    category="cost",
                    title="Enable Compression",
                    description="Use ENCODE AUTO for automatic compression",
                    current_pattern="CREATE TABLE without encoding",
                    suggested_pattern="CREATE TABLE {table} (...) ENCODE AUTO;",
                    estimated_impact="60-80% storage reduction",
                    platform="redshift"
                )
            },
            {
                "name": "materialized_view_candidate",
                "pattern": r"WITH\s+\w+\s+AS\s*\([^)]+GROUP\s+BY[^)]+\)",
                "check": self._check_materialized_view_candidate,
                "suggestion": OptimizationSuggestion(
                    severity="MEDIUM",
                    category="performance",
                    title="Consider Materialized View",
                    description="Complex aggregations could benefit from materialized views",
                    current_pattern="Heavy aggregation in CTE",
                    suggested_pattern="CREATE MATERIALIZED VIEW {view_name} AS ...",
                    estimated_impact="70-90% faster for repeated queries",
                    platform="redshift"
                )
            },
            {
                "name": "vacuum_analyze_needed",
                "pattern": r"DELETE|UPDATE",
                "check": self._check_vacuum_needed,
                "suggestion": OptimizationSuggestion(
                    severity="MEDIUM",
                    category="maintenance",
                    title="Schedule VACUUM and ANALYZE",
                    description="Tables with DELETE/UPDATE operations need regular VACUUM",
                    current_pattern="DELETE/UPDATE without VACUUM strategy",
                    suggested_pattern="VACUUM FULL {table}; ANALYZE {table};",
                    estimated_impact="Prevents query performance degradation",
                    platform="redshift"
                )
            }
        ]

    def _load_snowflake_rules(self) -> List[Dict]:
        """Load Snowflake-specific optimization rules"""
        return [
            {
                "name": "clustering_key_optimization",
                "pattern": r"WHERE\s+(\w+)\s*(?:=|IN\s*\()",
                "check": self._check_clustering_key,
                "suggestion": OptimizationSuggestion(
                    severity="HIGH",
                    category="performance",
                    title="Add Clustering Key",
                    description="Large tables benefit from clustering on filter columns",
                    current_pattern="Large table without clustering",
                    suggested_pattern="ALTER TABLE {table} CLUSTER BY ({columns});",
                    estimated_impact="50-80% reduction in scanned data",
                    platform="snowflake"
                )
            },
            {
                "name": "warehouse_sizing",
                "pattern": r"(?:COUNT|SUM|AVG|MAX|MIN)\s*\([^)]+\).*GROUP\s+BY",
                "check": self._check_warehouse_size,
                "suggestion": OptimizationSuggestion(
                    severity="MEDIUM",
                    category="cost",
                    title="Optimize Warehouse Size",
                    description="Use appropriate warehouse size for query complexity",
                    current_pattern="Complex query on small warehouse",
                    suggested_pattern="USE WAREHOUSE {appropriate_size}_WH;",
                    estimated_impact="Balance between performance and cost",
                    platform="snowflake"
                )
            },
            {
                "name": "result_caching",
                "pattern": r"SELECT",
                "check": self._check_result_caching,
                "suggestion": OptimizationSuggestion(
                    severity="LOW",
                    category="performance",
                    title="Enable Result Caching",
                    description="Leverage Snowflake's automatic result caching",
                    current_pattern="Repeated identical queries",
                    suggested_pattern="ALTER SESSION SET USE_CACHED_RESULT = TRUE;",
                    estimated_impact="Near-instant results for repeated queries",
                    platform="snowflake"
                )
            },
            {
                "name": "time_travel_optimization",
                "pattern": r"AT\s*\(|BEFORE\s*\(",
                "check": self._check_time_travel,
                "suggestion": OptimizationSuggestion(
                    severity="LOW",
                    category="cost",
                    title="Optimize Time Travel Retention",
                    description="Adjust DATA_RETENTION_TIME_IN_DAYS based on needs",
                    current_pattern="Default retention period",
                    suggested_pattern="ALTER TABLE {table} SET DATA_RETENTION_TIME_IN_DAYS = {days};",
                    estimated_impact="Reduce storage costs",
                    platform="snowflake"
                )
            },
            {
                "name": "semi_structured_optimization",
                "pattern": r"PARSE_JSON|FLATTEN|GET_PATH",
                "check": self._check_semi_structured,
                "suggestion": OptimizationSuggestion(
                    severity="MEDIUM",
                    category="performance",
                    title="Optimize Semi-Structured Data",
                    description="Consider materialized views for frequently accessed JSON paths",
                    current_pattern="Repeated JSON parsing",
                    suggested_pattern="CREATE MATERIALIZED VIEW ... AS SELECT GET_PATH(...)",
                    estimated_impact="70-90% faster JSON queries",
                    platform="snowflake"
                )
            }
        ]

    def _load_common_rules(self) -> List[Dict]:
        """Load platform-agnostic optimization rules"""
        return [
            {
                "name": "avoid_select_star",
                "pattern": r"SELECT\s+\*\s+FROM",
                "check": lambda sql: True,
                "suggestion": OptimizationSuggestion(
                    severity="MEDIUM",
                    category="performance",
                    title="Avoid SELECT *",
                    description="Explicitly list needed columns instead of SELECT *",
                    current_pattern="SELECT * FROM table",
                    suggested_pattern="SELECT col1, col2, col3 FROM table",
                    estimated_impact="Reduce data transfer and improve performance",
                    platform="both"
                )
            },
            {
                "name": "optimize_window_functions",
                "pattern": r"ROW_NUMBER\(\)|RANK\(\)|DENSE_RANK\(\)",
                "check": self._check_window_function,
                "suggestion": OptimizationSuggestion(
                    severity="MEDIUM",
                    category="performance",
                    title="Optimize Window Functions",
                    description="Consider materializing window function results",
                    current_pattern="Complex window functions in subqueries",
                    suggested_pattern="Create staging model with pre-calculated ranks",
                    estimated_impact="30-50% faster for complex queries",
                    platform="both"
                )
            },
            {
                "name": "cte_vs_subquery",
                "pattern": r"WITH.*?AS\s*\([^)]+\).*?WITH.*?AS\s*\([^)]+\)",
                "check": self._check_multiple_ctes,
                "suggestion": OptimizationSuggestion(
                    severity="LOW",
                    category="maintenance",
                    title="Consider Staging Models",
                    description="Multiple CTEs might benefit from staging models",
                    current_pattern="Many CTEs in single query",
                    suggested_pattern="Break into separate staging models",
                    estimated_impact="Better maintainability and reusability",
                    platform="both"
                )
            },
            {
                "name": "union_vs_union_all",
                "pattern": r"UNION(?!\s+ALL)",
                "check": lambda sql: True,
                "suggestion": OptimizationSuggestion(
                    severity="MEDIUM",
                    category="performance",
                    title="Use UNION ALL When Possible",
                    description="UNION ALL is faster when duplicates don't need removal",
                    current_pattern="UNION",
                    suggested_pattern="UNION ALL",
                    estimated_impact="20-40% faster for large datasets",
                    platform="both"
                )
            },
            {
                "name": "incremental_model_candidate",
                "pattern": r"INSERT|MERGE|DELETE.*WHERE",
                "check": self._check_incremental_candidate,
                "suggestion": OptimizationSuggestion(
                    severity="HIGH",
                    category="performance",
                    title="Consider Incremental Model",
                    description="Large tables with date filters should be incremental",
                    current_pattern="Full table refresh",
                    suggested_pattern="{{ config(materialized='incremental') }}",
                    estimated_impact="80-95% reduction in processing time",
                    platform="both"
                )
            }
        ]

    def analyze_sql(self, sql: str, platform: str = "redshift") -> List[OptimizationSuggestion]:
        """
        Analyze SQL and return optimization suggestions
        """
        suggestions = []
        sql_upper = sql.upper()

        # Check platform-specific rules
        if platform.lower() == "redshift":
            rules = self.redshift_rules
        elif platform.lower() == "snowflake":
            rules = self.snowflake_rules
        else:
            rules = []

        # Add common rules
        rules.extend(self.common_rules)

        # Check each rule
        for rule in rules:
            if re.search(rule["pattern"], sql_upper):
                if rule["check"](sql):
                    suggestion = rule["suggestion"]
                    # Skip if platform doesn't match
                    if suggestion.platform != "both" and suggestion.platform != platform.lower():
                        continue
                    suggestions.append(suggestion)

        # Sort by severity
        severity_order = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}
        suggestions.sort(key=lambda x: severity_order.get(x.severity, 4))

        return suggestions

    def _check_distribution_key(self, sql: str) -> bool:
        """Check if distribution key optimization is needed"""
        # Check for large table joins without DISTKEY hint
        return "JOIN" in sql.upper() and "DISTKEY" not in sql.upper()

    def _check_sort_key(self, sql: str) -> bool:
        """Check if sort key optimization is needed"""
        # Check for WHERE clauses without SORTKEY hint
        return "WHERE" in sql.upper() and "SORTKEY" not in sql.upper()

    def _check_compression(self, sql: str) -> bool:
        """Check if compression encoding is needed"""
        return "CREATE TABLE" in sql.upper() and "ENCODE" not in sql.upper()

    def _check_materialized_view_candidate(self, sql: str) -> bool:
        """Check if query is a good materialized view candidate"""
        # Complex aggregations in CTEs
        return "GROUP BY" in sql.upper() and "WITH" in sql.upper()

    def _check_vacuum_needed(self, sql: str) -> bool:
        """Check if VACUUM is needed"""
        return "DELETE" in sql.upper() or "UPDATE" in sql.upper()

    def _check_clustering_key(self, sql: str) -> bool:
        """Check if clustering key is needed"""
        return "WHERE" in sql.upper() and ("=" in sql or "IN" in sql.upper())

    def _check_warehouse_size(self, sql: str) -> bool:
        """Check if warehouse sizing needs optimization"""
        # Complex aggregations might need larger warehouse
        agg_count = sum(1 for agg in ["COUNT", "SUM", "AVG", "MAX", "MIN"] if agg in sql.upper())
        return agg_count > 2

    def _check_result_caching(self, sql: str) -> bool:
        """Check if result caching would help"""
        # Always suggest for SELECT queries
        return "SELECT" in sql.upper()

    def _check_time_travel(self, sql: str) -> bool:
        """Check if time travel is being used"""
        return "AT(" in sql.upper() or "BEFORE(" in sql.upper()

    def _check_semi_structured(self, sql: str) -> bool:
        """Check if semi-structured data needs optimization"""
        json_functions = ["PARSE_JSON", "FLATTEN", "GET_PATH", "JSON_EXTRACT"]
        return any(func in sql.upper() for func in json_functions)

    def _check_window_function(self, sql: str) -> bool:
        """Check if window functions need optimization"""
        window_functions = ["ROW_NUMBER()", "RANK()", "DENSE_RANK()", "LAG(", "LEAD("]
        return any(func in sql.upper() for func in window_functions)

    def _check_multiple_ctes(self, sql: str) -> bool:
        """Check if multiple CTEs exist"""
        return sql.upper().count("WITH") > 1 or sql.upper().count("AS (") > 2

    def _check_incremental_candidate(self, sql: str) -> bool:
        """Check if model is a good incremental candidate"""
        # Look for date filters or merge patterns
        date_patterns = ["DATE", "TIMESTAMP", "CREATED_AT", "UPDATED_AT"]
        return any(pattern in sql.upper() for pattern in date_patterns)

    def generate_optimization_report(
        self,
        sql: str,
        model_name: str,
        platform: str = "redshift"
    ) -> str:
        """Generate a detailed optimization report"""
        suggestions = self.analyze_sql(sql, platform)

        report = f"""
# Optimization Report for {model_name}
**Platform**: {platform.capitalize()}
**Analysis Date**: {__import__('datetime').datetime.now().isoformat()}

## Summary
Found **{len(suggestions)}** optimization opportunities

## Suggestions by Priority
"""

        # Group by severity
        by_severity = {}
        for suggestion in suggestions:
            if suggestion.severity not in by_severity:
                by_severity[suggestion.severity] = []
            by_severity[suggestion.severity].append(suggestion)

        for severity in ["CRITICAL", "HIGH", "MEDIUM", "LOW"]:
            if severity in by_severity:
                report += f"\n### {severity} Priority\n"
                for i, suggestion in enumerate(by_severity[severity], 1):
                    report += f"""
{i}. **{suggestion.title}**
   - *Category*: {suggestion.category}
   - *Description*: {suggestion.description}
   - *Current Pattern*: `{suggestion.current_pattern}`
   - *Suggested Pattern*: `{suggestion.suggested_pattern}`
   - *Estimated Impact*: {suggestion.estimated_impact}
"""

        # Add implementation guide
        report += """

## Implementation Guide

1. **Start with CRITICAL and HIGH priority items** - These offer the most significant improvements
2. **Test changes in development** - Always validate optimizations before production
3. **Monitor performance metrics** - Track query execution time and resource usage
4. **Consider incremental implementation** - Apply changes gradually to minimize risk

## Platform-Specific Notes
"""

        if platform.lower() == "redshift":
            report += """
### Redshift Best Practices
- Run ANALYZE after significant data changes
- Schedule regular VACUUM operations
- Monitor WLM queue performance
- Use Redshift Advisor recommendations
"""
        elif platform.lower() == "snowflake":
            report += """
### Snowflake Best Practices
- Monitor warehouse utilization
- Use auto-suspend and auto-resume
- Leverage result caching
- Consider Snowpipe for continuous loading
"""

        return report

    def export_suggestions_json(
        self,
        suggestions: List[OptimizationSuggestion]
    ) -> str:
        """Export suggestions as JSON for programmatic use"""
        return json.dumps(
            [
                {
                    "severity": s.severity,
                    "category": s.category,
                    "title": s.title,
                    "description": s.description,
                    "current_pattern": s.current_pattern,
                    "suggested_pattern": s.suggested_pattern,
                    "estimated_impact": s.estimated_impact,
                    "platform": s.platform
                }
                for s in suggestions
            ],
            indent=2
        )