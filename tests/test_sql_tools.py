"""
Comprehensive tests for SQL analysis tools
"""
import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from adam.tools.sql_tools import (
    SQLAnalyzer, SQLFormatter, SQLOptimizer, SQLIssue, 
    IssueLevel, QueryMetrics, analyze_sql
)


class TestSQLAnalyzer:
    """Test SQL analysis functionality"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.analyzer = SQLAnalyzer("snowflake")
        
    def test_select_star_detection(self):
        """Test detection of SELECT * anti-pattern"""
        query = """
        SELECT * 
        FROM customers 
        WHERE created_date > '2024-01-01'
        """
        issues, metrics = self.analyzer.analyze_query(query)
        
        # Should find SELECT * issue
        select_star_issues = [i for i in issues if "SELECT *" in i.message]
        assert len(select_star_issues) == 1
        assert select_star_issues[0].level == IssueLevel.WARNING
        assert select_star_issues[0].line_number == 2
        
    def test_missing_where_clause(self):
        """Test detection of dangerous DELETE/UPDATE without WHERE"""
        # Test DELETE without WHERE
        delete_query = "DELETE FROM users"
        issues, _ = self.analyzer.analyze_query(delete_query)
        
        delete_issues = [i for i in issues if "DELETE without WHERE" in i.message]
        assert len(delete_issues) == 1
        assert delete_issues[0].level == IssueLevel.ERROR
        
        # Test UPDATE without WHERE
        update_query = "UPDATE users SET active = false"
        issues, _ = self.analyzer.analyze_query(update_query)
        
        update_issues = [i for i in issues if "UPDATE without WHERE" in i.message]
        assert len(update_issues) == 1
        assert update_issues[0].level == IssueLevel.ERROR
        
    def test_implicit_cross_join_detection(self):
        """Test detection of implicit cross joins"""
        query = """
        SELECT *
        FROM orders, customers
        WHERE orders.customer_id = customers.id
        """
        issues, _ = self.analyzer.analyze_query(query)
        
        cross_join_issues = [i for i in issues if "Implicit cross join" in i.message]
        assert len(cross_join_issues) == 1
        assert cross_join_issues[0].level == IssueLevel.WARNING
        
    def test_expensive_operations(self):
        """Test detection of expensive SQL operations"""
        # Leading wildcard
        query1 = "SELECT * FROM products WHERE name LIKE '%phone'"
        issues, _ = self.analyzer.analyze_query(query1)
        assert any("leading wildcard" in i.message for i in issues)
        
        # NOT IN with subquery
        query2 = """
        SELECT * FROM orders 
        WHERE customer_id NOT IN (SELECT id FROM banned_customers)
        """
        issues, _ = self.analyzer.analyze_query(query2)
        assert any("NOT IN with subquery" in i.message for i in issues)
        
        # OR in JOIN
        query3 = """
        SELECT * FROM orders o
        JOIN customers c ON o.customer_id = c.id OR o.guest_id = c.guest_id
        """
        issues, _ = self.analyzer.analyze_query(query3)
        assert any("OR in JOIN condition" in i.message for i in issues)
        
    def test_subquery_anti_patterns(self):
        """Test detection of subquery issues"""
        # Multiple subqueries in SELECT
        query = """
        SELECT 
            customer_id,
            (SELECT COUNT(*) FROM orders WHERE customer_id = c.id) as order_count,
            (SELECT SUM(amount) FROM orders WHERE customer_id = c.id) as total_spent,
            (SELECT MAX(date) FROM orders WHERE customer_id = c.id) as last_order
        FROM customers c
        """
        issues, _ = self.analyzer.analyze_query(query)
        
        subquery_issues = [i for i in issues if "Multiple subqueries" in i.message]
        assert len(subquery_issues) == 1
        
        # Correlated subquery
        correlated_issues = [i for i in issues if "correlated subquery" in i.message]
        assert len(correlated_issues) == 1
        
    def test_query_metrics(self):
        """Test query complexity metrics calculation"""
        complex_query = """
        WITH 
        base_orders AS (
            SELECT * FROM orders WHERE date > '2024-01-01'
        ),
        customer_summary AS (
            SELECT 
                customer_id,
                COUNT(DISTINCT order_id) as order_count
            FROM base_orders
            GROUP BY customer_id
        ),
        product_summary AS (
            SELECT 
                product_id,
                SUM(amount) as total_revenue
            FROM base_orders
            GROUP BY product_id
        )
        SELECT DISTINCT
            c.customer_id,
            c.order_count,
            p.total_revenue
        FROM customer_summary c
        LEFT JOIN product_summary p ON c.customer_id = p.product_id
        UNION ALL
        SELECT DISTINCT
            customer_id,
            0 as order_count,
            0 as total_revenue
        FROM archived_customers
        """
        
        issues, metrics = self.analyzer.analyze_query(complex_query)
        
        assert metrics.cte_count >= 3
        assert metrics.join_count >= 1
        assert metrics.distinct_count >= 2
        assert metrics.complexity_score >= 3
        
    def test_snowflake_specific_checks(self):
        """Test Snowflake-specific optimizations"""
        analyzer = SQLAnalyzer("snowflake")
        
        # Date filtering without clustering hint
        query = """
        SELECT * FROM events 
        WHERE event_date BETWEEN '2024-01-01' AND '2024-01-31'
        """
        issues, _ = analyzer.analyze_query(query)
        
        clustering_issues = [i for i in issues if "clustering" in i.message.lower()]
        assert len(clustering_issues) >= 1
        
        # FLATTEN without LATERAL
        query2 = """
        SELECT * FROM orders, 
        FLATTEN(order_items) as items
        """
        issues, _ = analyzer.analyze_query(query2)
        
        flatten_issues = [i for i in issues if "FLATTEN" in i.message and "LATERAL" in i.message]
        assert len(flatten_issues) == 1
        
    def test_cte_usage_analysis(self):
        """Test CTE usage pattern detection"""
        # Unused CTE
        query = """
        WITH unused_cte AS (
            SELECT * FROM orders
        ),
        used_cte AS (
            SELECT * FROM customers
        )
        SELECT * FROM used_cte
        """
        issues, _ = self.analyzer.analyze_query(query)
        
        unused_cte_issues = [i for i in issues if "never used" in i.message]
        assert len(unused_cte_issues) == 1
        assert "unused_cte" in unused_cte_issues[0].message


class TestSQLFormatter:
    """Test SQL formatting functionality"""
    
    def test_dbt_style_formatting(self):
        """Test dbt SQL style formatting"""
        formatter = SQLFormatter("dbt")
        
        query = "SELECT customer_id, order_date, amount FROM orders WHERE status = 'completed'"
        formatted = formatter.format_query(query)
        
        # Should be multi-line
        assert '\n' in formatted
        # Keywords should be uppercase
        assert 'SELECT' in formatted
        assert 'FROM' in formatted
        assert 'WHERE' in formatted
        
    def test_cte_formatting(self):
        """Test CTE formatting"""
        formatter = SQLFormatter("dbt")
        
        query = """WITH orders AS (SELECT * FROM raw_orders), customers AS (SELECT * FROM raw_customers) SELECT * FROM orders"""
        formatted = formatter.format_query(query)
        
        # CTEs should be on separate lines
        assert 'WITH\n' in formatted or 'WITH ' in formatted
        # Each CTE should be clearly separated
        lines = formatted.split('\n')
        assert len(lines) > 3
        
    def test_syntax_validation(self):
        """Test SQL syntax validation"""
        formatter = SQLFormatter()
        
        # Valid query
        errors = formatter.validate_syntax("SELECT * FROM users")
        assert len(errors) == 0
        
        # Unbalanced parentheses
        errors = formatter.validate_syntax("SELECT * FROM users WHERE id IN (1, 2, 3")
        assert any("parentheses" in e for e in errors)
        
        # Unbalanced quotes
        errors = formatter.validate_syntax("SELECT * FROM users WHERE name = 'John")
        assert any("quotes" in e for e in errors)
        
        # Common typos
        errors = formatter.validate_syntax("SELECT * FORM users")
        assert any("FORM" in e and "FROM" in e for e in errors)
        
        errors = formatter.validate_syntax("SELECT * FROM users WEHRE id = 1")
        assert any("WEHRE" in e and "WHERE" in e for e in errors)


class TestSQLOptimizer:
    """Test SQL optimization functionality"""
    
    @pytest.mark.asyncio
    async def test_optimize_query_basic(self):
        """Test basic query optimization"""
        optimizer = SQLOptimizer("snowflake")
        
        # Mock the LLM client
        with patch.object(optimizer.analyzer, '_get_llm_client') as mock_llm:
            mock_client = AsyncMock()
            mock_response = Mock()
            mock_response.content = """
SELECT 
    customer_id,
    order_date,
    amount
FROM orders
WHERE status = 'completed'
-- Added specific column selection instead of SELECT *
"""
            mock_client.complete.return_value = mock_response
            mock_llm.return_value = mock_client
            
            query = "SELECT * FROM orders WHERE status = 'completed'"
            result = await optimizer.optimize_query(query)
            
            assert 'optimized_query' in result
            assert 'issues' in result
            assert 'metrics' in result
            assert 'recommendations' in result
            assert len(result['issues']) > 0  # Should find SELECT * issue
            
    @pytest.mark.asyncio
    async def test_complex_optimization(self):
        """Test optimization of complex query"""
        optimizer = SQLOptimizer("snowflake")
        
        complex_query = """
        SELECT DISTINCT *
        FROM orders o, customers c, products p
        WHERE o.customer_id = c.id 
        AND o.product_id = p.id
        AND o.amount NOT IN (SELECT amount FROM cancelled_orders)
        AND o.created_date > '2024-01-01'
        """
        
        # Mock LLM
        with patch.object(optimizer.analyzer, '_get_llm_client') as mock_llm:
            mock_client = AsyncMock()
            mock_response = Mock()
            mock_response.content = """
SELECT DISTINCT
    o.order_id,
    o.customer_id,
    c.customer_name,
    p.product_name,
    o.amount,
    o.created_date
FROM orders o
JOIN customers c ON o.customer_id = c.id
JOIN products p ON o.product_id = p.id
WHERE NOT EXISTS (
    SELECT 1 FROM cancelled_orders co 
    WHERE co.amount = o.amount
)
AND o.created_date > '2024-01-01'
-- Improvements:
-- 1. Replaced implicit joins with explicit JOIN syntax
-- 2. Replaced SELECT * with specific columns
-- 3. Replaced NOT IN with NOT EXISTS for better NULL handling
"""
            mock_client.complete.return_value = mock_response
            mock_llm.return_value = mock_client
            
            result = await optimizer.optimize_query(complex_query)
            
            # Should find multiple issues
            assert len(result['issues']) >= 3
            assert any(i.level == IssueLevel.WARNING for i in result['issues'])
            
            # Should have recommendations
            assert len(result['recommendations']) > 0
            
            # Should estimate improvement
            assert "improvement" in result['estimated_improvement'].lower()


class TestConvenienceFunctions:
    """Test convenience functions"""
    
    @pytest.mark.asyncio
    async def test_analyze_sql_function(self):
        """Test the analyze_sql convenience function"""
        with patch('adam.tools.sql_tools.SQLOptimizer.optimize_query') as mock_optimize:
            mock_optimize.return_value = {
                'optimized_query': 'SELECT id FROM users',
                'issues': [],
                'metrics': Mock(),
                'recommendations': []
            }
            
            result = await analyze_sql("SELECT * FROM users")
            assert mock_optimize.called


def test_issue_levels():
    """Test issue level enum"""
    assert IssueLevel.ERROR.value == "error"
    assert IssueLevel.WARNING.value == "warning"
    assert IssueLevel.SUGGESTION.value == "suggestion"
    assert IssueLevel.INFO.value == "info"


def test_sql_issue_dataclass():
    """Test SQLIssue dataclass"""
    issue = SQLIssue(
        level=IssueLevel.WARNING,
        message="Test issue",
        line_number=5,
        suggestion="Fix it",
        estimated_impact="High"
    )
    
    assert issue.level == IssueLevel.WARNING
    assert issue.message == "Test issue"
    assert issue.line_number == 5
    assert issue.suggestion == "Fix it"
    assert issue.estimated_impact == "High"


def test_query_metrics_dataclass():
    """Test QueryMetrics dataclass"""
    metrics = QueryMetrics(
        line_count=50,
        cte_count=3,
        join_count=2,
        subquery_count=1,
        distinct_count=1,
        complexity_score=5
    )
    
    assert metrics.line_count == 50
    assert metrics.complexity_score == 5


if __name__ == "__main__":
    # Run specific test for debugging
    test = TestSQLAnalyzer()
    test.setup_method()
    test.test_select_star_detection()
    print("✅ Basic tests pass")
    
    # Run async test
    async def run_async_test():
        test = TestSQLOptimizer()
        await test.test_optimize_query_basic()
        print("✅ Async tests pass")
        
    asyncio.run(run_async_test())