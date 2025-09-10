"""
Snowflake SQL Executor for ADAM
Execute SQL queries against Snowflake data warehouse
"""

import os
import json
import logging
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass
from datetime import datetime
import pandas as pd
from io import StringIO

logger = logging.getLogger(__name__)

# Try to import Snowflake connector
try:
    import snowflake.connector
    from snowflake.connector import DictCursor
    SNOWFLAKE_AVAILABLE = True
except ImportError:
    SNOWFLAKE_AVAILABLE = False
    logger.warning("Snowflake connector not installed. Run: pip install snowflake-connector-python")

# For data visualization
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False
    logger.info("Plotting libraries not available. Run: pip install matplotlib seaborn")


@dataclass
class QueryResult:
    """Result from Snowflake query execution"""
    query: str
    success: bool
    data: Optional[pd.DataFrame] = None
    row_count: int = 0
    column_names: List[str] = None
    execution_time: float = 0.0
    error: Optional[str] = None
    query_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        result = {
            'query': self.query,
            'success': self.success,
            'row_count': self.row_count,
            'column_names': self.column_names,
            'execution_time': self.execution_time,
            'error': self.error,
            'query_id': self.query_id
        }
        
        if self.data is not None and not self.data.empty:
            result['data_preview'] = self.data.head(10).to_dict('records')
            result['data_shape'] = {'rows': len(self.data), 'columns': len(self.data.columns)}
        
        return result
    
    def to_csv(self) -> str:
        """Export results to CSV"""
        if self.data is not None:
            return self.data.to_csv(index=False)
        return ""
    
    def to_json(self) -> str:
        """Export results to JSON"""
        if self.data is not None:
            return self.data.to_json(orient='records', indent=2)
        return "[]"


class SnowflakeExecutor:
    """
    Execute SQL queries against Snowflake with safety features
    """
    
    def __init__(self, connection_params: Dict[str, str] = None):
        """
        Initialize Snowflake executor
        
        Args:
            connection_params: Dict with account, user, password, warehouse, database, schema
                              If None, will read from environment variables
        """
        if not SNOWFLAKE_AVAILABLE:
            raise ImportError("Snowflake connector required. Install with: pip install snowflake-connector-python")
        
        self.connection_params = connection_params or self._get_env_params()
        self.connection = None
        self.query_history = []
        
    def _get_env_params(self) -> Dict[str, str]:
        """Get Snowflake connection params from environment"""
        return {
            'account': os.getenv('SNOWFLAKE_ACCOUNT'),
            'user': os.getenv('SNOWFLAKE_USER'),
            'password': os.getenv('SNOWFLAKE_PASSWORD'),
            'warehouse': os.getenv('SNOWFLAKE_WAREHOUSE', 'COMPUTE_WH'),
            'database': os.getenv('SNOWFLAKE_DATABASE'),
            'schema': os.getenv('SNOWFLAKE_SCHEMA', 'PUBLIC'),
            'role': os.getenv('SNOWFLAKE_ROLE')
        }
    
    def connect(self) -> bool:
        """Establish connection to Snowflake"""
        try:
            self.connection = snowflake.connector.connect(
                **{k: v for k, v in self.connection_params.items() if v}
            )
            logger.info(f"Connected to Snowflake account: {self.connection_params['account']}")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to Snowflake: {e}")
            return False
    
    def disconnect(self):
        """Close Snowflake connection"""
        if self.connection:
            self.connection.close()
            self.connection = None
    
    def validate_query(self, query: str) -> tuple[bool, str]:
        """
        Validate SQL query for safety
        
        Returns:
            (is_safe, message)
        """
        query_upper = query.upper()
        
        # Check for dangerous operations
        dangerous_keywords = ['DROP', 'TRUNCATE', 'DELETE', 'ALTER', 'CREATE USER', 'GRANT']
        for keyword in dangerous_keywords:
            if keyword in query_upper:
                return False, f"Query contains potentially dangerous operation: {keyword}"
        
        # Check for read-only operations (configurable)
        if os.getenv('SNOWFLAKE_READONLY', 'false').lower() == 'true':
            write_keywords = ['INSERT', 'UPDATE', 'CREATE', 'MERGE', 'COPY']
            for keyword in write_keywords:
                if keyword in query_upper:
                    return False, f"Read-only mode: {keyword} operations not allowed"
        
        return True, "Query validated"
    
    def execute(self, query: str, fetch_results: bool = True,
                max_rows: int = 10000, safe_mode: bool = True) -> QueryResult:
        """
        Execute SQL query against Snowflake
        
        Args:
            query: SQL query to execute
            fetch_results: Whether to fetch and return results
            max_rows: Maximum rows to fetch
            safe_mode: Whether to validate query for safety
            
        Returns:
            QueryResult object
        """
        start_time = datetime.now()
        
        # Validate query if in safe mode
        if safe_mode:
            is_safe, message = self.validate_query(query)
            if not is_safe:
                return QueryResult(
                    query=query,
                    success=False,
                    error=message,
                    execution_time=0
                )
        
        # Ensure connection
        if not self.connection:
            if not self.connect():
                return QueryResult(
                    query=query,
                    success=False,
                    error="Failed to connect to Snowflake",
                    execution_time=0
                )
        
        try:
            cursor = self.connection.cursor(DictCursor)
            
            # Execute query
            cursor.execute(query)
            query_id = cursor.sfqid
            
            # Fetch results if requested
            data = None
            column_names = []
            row_count = 0
            
            if fetch_results and cursor.description:
                # Get column names
                column_names = [col[0] for col in cursor.description]
                
                # Fetch data
                if max_rows > 0:
                    rows = cursor.fetchmany(max_rows)
                else:
                    rows = cursor.fetchall()
                
                # Convert to DataFrame
                if rows:
                    data = pd.DataFrame(rows)
                    row_count = len(data)
                else:
                    data = pd.DataFrame(columns=column_names)
            
            cursor.close()
            
            # Calculate execution time
            execution_time = (datetime.now() - start_time).total_seconds()
            
            # Store in history
            result = QueryResult(
                query=query,
                success=True,
                data=data,
                row_count=row_count,
                column_names=column_names,
                execution_time=execution_time,
                query_id=query_id
            )
            
            self.query_history.append(result)
            return result
            
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            error_msg = str(e)
            
            return QueryResult(
                query=query,
                success=False,
                error=error_msg,
                execution_time=execution_time
            )
    
    def execute_many(self, queries: List[str]) -> List[QueryResult]:
        """Execute multiple queries in sequence"""
        results = []
        for query in queries:
            result = self.execute(query)
            results.append(result)
            if not result.success:
                logger.warning(f"Query failed, stopping batch execution: {result.error}")
                break
        return results
    
    def get_table_info(self, table_name: str) -> QueryResult:
        """Get information about a table"""
        query = f"""
        SELECT 
            column_name,
            data_type,
            is_nullable,
            column_default,
            comment
        FROM information_schema.columns
        WHERE table_name = '{table_name.upper()}'
        ORDER BY ordinal_position
        """
        return self.execute(query)
    
    def get_table_sample(self, table_name: str, sample_size: int = 100) -> QueryResult:
        """Get a sample of data from a table"""
        query = f"SELECT * FROM {table_name} SAMPLE ({sample_size} ROWS)"
        return self.execute(query)
    
    def analyze_query_plan(self, query: str) -> QueryResult:
        """Get query execution plan"""
        explain_query = f"EXPLAIN {query}"
        return self.execute(explain_query)
    
    def get_query_history(self, last_n: int = 10) -> List[Dict[str, Any]]:
        """Get recent query history"""
        history = []
        for result in self.query_history[-last_n:]:
            history.append({
                'query': result.query[:100] + '...' if len(result.query) > 100 else result.query,
                'success': result.success,
                'row_count': result.row_count,
                'execution_time': result.execution_time,
                'error': result.error
            })
        return history


class SnowflakeQueryBuilder:
    """
    Helper class to build common Snowflake queries
    """
    
    @staticmethod
    def select(table: str, columns: List[str] = None, where: str = None,
               order_by: str = None, limit: int = None) -> str:
        """Build SELECT query"""
        cols = ', '.join(columns) if columns else '*'
        query = f"SELECT {cols} FROM {table}"
        
        if where:
            query += f" WHERE {where}"
        if order_by:
            query += f" ORDER BY {order_by}"
        if limit:
            query += f" LIMIT {limit}"
        
        return query
    
    @staticmethod
    def aggregate(table: str, group_by: List[str], aggregates: Dict[str, str],
                  where: str = None, having: str = None) -> str:
        """
        Build aggregation query
        
        Args:
            table: Table name
            group_by: Columns to group by
            aggregates: Dict of {alias: aggregate_expression}
            where: WHERE clause
            having: HAVING clause
        """
        agg_list = [f"{expr} AS {alias}" for alias, expr in aggregates.items()]
        
        query = f"SELECT {', '.join(group_by)}, {', '.join(agg_list)} FROM {table}"
        
        if where:
            query += f" WHERE {where}"
        
        query += f" GROUP BY {', '.join(group_by)}"
        
        if having:
            query += f" HAVING {having}"
        
        return query
    
    @staticmethod
    def join(left_table: str, right_table: str, join_type: str = "INNER",
             on_condition: str = None, using_columns: List[str] = None) -> str:
        """Build JOIN query"""
        query = f"SELECT * FROM {left_table} {join_type} JOIN {right_table}"
        
        if on_condition:
            query += f" ON {on_condition}"
        elif using_columns:
            query += f" USING ({', '.join(using_columns)})"
        
        return query
    
    @staticmethod
    def window_function(table: str, columns: List[str], 
                       window_functions: Dict[str, str],
                       partition_by: List[str] = None,
                       order_by: str = None) -> str:
        """
        Build query with window functions
        
        Args:
            table: Table name
            columns: Regular columns to select
            window_functions: Dict of {alias: window_expression}
            partition_by: Columns for PARTITION BY
            order_by: ORDER BY for window
        """
        cols = ', '.join(columns) if columns else '*'
        
        window_clause = "OVER ("
        if partition_by:
            window_clause += f"PARTITION BY {', '.join(partition_by)} "
        if order_by:
            window_clause += f"ORDER BY {order_by}"
        window_clause += ")"
        
        window_exprs = [f"{expr} {window_clause} AS {alias}" 
                       for alias, expr in window_functions.items()]
        
        all_selections = [cols] + window_exprs
        
        return f"SELECT {', '.join(all_selections)} FROM {table}"


class SnowflakeDataVisualizer:
    """
    Visualize query results
    """
    
    @staticmethod
    def plot_distribution(data: pd.DataFrame, column: str, 
                         title: str = None) -> Optional[plt.Figure]:
        """Plot distribution of a column"""
        if not PLOTTING_AVAILABLE:
            logger.warning("Plotting libraries not available")
            return None
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        if data[column].dtype in ['int64', 'float64']:
            data[column].hist(ax=ax, bins=30)
            ax.set_xlabel(column)
            ax.set_ylabel('Frequency')
        else:
            data[column].value_counts().head(20).plot(kind='bar', ax=ax)
            ax.set_xlabel(column)
            ax.set_ylabel('Count')
        
        ax.set_title(title or f'Distribution of {column}')
        plt.tight_layout()
        return fig
    
    @staticmethod
    def plot_time_series(data: pd.DataFrame, date_column: str,
                        value_columns: List[str], title: str = None) -> Optional[plt.Figure]:
        """Plot time series data"""
        if not PLOTTING_AVAILABLE:
            return None
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        data[date_column] = pd.to_datetime(data[date_column])
        data = data.sort_values(date_column)
        
        for col in value_columns:
            ax.plot(data[date_column], data[col], label=col, marker='o')
        
        ax.set_xlabel('Date')
        ax.set_ylabel('Value')
        ax.set_title(title or 'Time Series Plot')
        ax.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        return fig
    
    @staticmethod
    def create_summary_stats(data: pd.DataFrame) -> pd.DataFrame:
        """Generate summary statistics for numerical columns"""
        return data.describe()


# Convenience functions
def quick_query(query: str) -> pd.DataFrame:
    """
    Execute a quick query and return results as DataFrame
    
    Example:
        df = quick_query("SELECT * FROM sales LIMIT 100")
    """
    executor = SnowflakeExecutor()
    result = executor.execute(query)
    executor.disconnect()
    
    if result.success and result.data is not None:
        return result.data
    else:
        raise Exception(f"Query failed: {result.error}")


def analyze_table(table_name: str) -> Dict[str, Any]:
    """
    Get comprehensive analysis of a table
    
    Returns:
        Dict with schema, sample data, and statistics
    """
    executor = SnowflakeExecutor()
    
    # Get table info
    info_result = executor.get_table_info(table_name)
    
    # Get sample data
    sample_result = executor.get_table_sample(table_name, 100)
    
    # Get row count
    count_result = executor.execute(f"SELECT COUNT(*) as row_count FROM {table_name}")
    
    executor.disconnect()
    
    return {
        'schema': info_result.data.to_dict('records') if info_result.success else None,
        'sample_data': sample_result.data.head(10).to_dict('records') if sample_result.success else None,
        'row_count': count_result.data['row_count'][0] if count_result.success else None,
        'columns': info_result.column_names if info_result.success else None
    }


if __name__ == "__main__":
    # Demo usage
    print("Snowflake SQL Executor Demo")
    print("-" * 40)
    
    # Check if credentials are available
    if not os.getenv('SNOWFLAKE_ACCOUNT'):
        print("Please set SNOWFLAKE_ACCOUNT, SNOWFLAKE_USER, and SNOWFLAKE_PASSWORD environment variables")
        print("\nExample usage:")
        print("  executor = SnowflakeExecutor()")
        print("  result = executor.execute('SELECT * FROM my_table LIMIT 10')")
        print("  print(result.data)")
    else:
        # Try to connect and run a simple query
        executor = SnowflakeExecutor()
        if executor.connect():
            print("Connected to Snowflake successfully!")
            
            # Run a simple query
            result = executor.execute("SELECT CURRENT_TIMESTAMP() as current_time")
            if result.success:
                print(f"Current Snowflake time: {result.data['current_time'][0]}")
            
            executor.disconnect()
        else:
            print("Failed to connect to Snowflake")