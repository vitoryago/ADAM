# ADAM Snowflake Integration

## Overview

ADAM now has comprehensive Snowflake SQL capabilities:
- **Generate** optimized SQL queries with Snowflake-specific syntax
- **Execute** queries directly against Snowflake data warehouse
- **Analyze** results with built-in data quality checks
- **Visualize** data with matplotlib/seaborn integration

## Components Built

### 1. Snowflake Executor (`snowflake_executor.py`)
- Direct Snowflake connection and query execution
- Query validation for safety (prevents DROP, TRUNCATE, etc.)
- Result handling with pandas DataFrames
- Query history tracking
- Batch query execution

### 2. Enhanced File Generator
Added 5 new SQL templates:
- `sql_query` - Basic SELECT and aggregation queries
- `snowflake_query` - Snowflake-optimized with CTEs and window functions
- `sql_analysis` - Comprehensive data quality and analysis suite
- `dbt_model` - dbt transformation models
- `sql_stored_procedure` - Snowflake stored procedures

### 3. Query Builder
Programmatic query construction:
- SELECT with filters and ordering
- Aggregations with GROUP BY/HAVING
- JOINs (INNER, LEFT, RIGHT, FULL)
- Window functions with PARTITION BY

## Usage Examples

### Generate SQL Query
```python
from adam.tools import FileGenerator

generator = FileGenerator()
query = generator.generate(
    'snowflake_query',
    'sales_analysis',
    warehouse='ANALYTICS_WH',
    database='SALES_DB'
)
print(query.content)
```

### Execute on Snowflake
```python
from adam.tools.snowflake_executor import SnowflakeExecutor

executor = SnowflakeExecutor()
result = executor.execute("""
    SELECT 
        DATE_TRUNC('month', order_date) as month,
        SUM(amount) as revenue
    FROM orders
    GROUP BY 1
""")

if result.success:
    df = result.data  # pandas DataFrame
    print(df.head())
```

### Build Complex Queries
```python
from adam.tools.snowflake_executor import SnowflakeQueryBuilder

builder = SnowflakeQueryBuilder()
query = builder.window_function(
    table='sales',
    columns=['date', 'product'],
    window_functions={
        'rank': 'RANK()',
        'running_total': 'SUM(amount)'
    },
    partition_by=['product'],
    order_by='date'
)
```

## ADAM Conversation Examples

Users can now ask ADAM:

### Query Generation
- "Generate a Snowflake query to analyze customer churn"
- "Create a dbt model for daily sales aggregation"
- "Write a stored procedure for monthly reporting"

### Data Analysis
- "Analyze the orders table for data quality issues"
- "Show me the distribution of sales by category"
- "Find anomalies in the transaction data"

### Query Optimization
- "Optimize this query for Snowflake"
- "Add proper indexing hints"
- "Convert this to use window functions"

## Environment Setup

### Required for Execution
```bash
pip install snowflake-connector-python pandas

# Set environment variables
export SNOWFLAKE_ACCOUNT=your_account
export SNOWFLAKE_USER=your_user
export SNOWFLAKE_PASSWORD=your_password
export SNOWFLAKE_DATABASE=your_database
export SNOWFLAKE_WAREHOUSE=your_warehouse
```

### Optional for Visualization
```bash
pip install matplotlib seaborn
```

## Safety Features

### Query Validation
- Blocks dangerous operations (DROP, TRUNCATE, DELETE)
- Read-only mode option via `SNOWFLAKE_READONLY=true`
- Validates syntax before execution

### Resource Limits
- Max rows limit (default: 10,000)
- Query timeout protection
- Connection pooling

## Architecture Benefits

### Why This Approach?
1. **ADAM generates the SQL** - Uses LLM to write optimized queries
2. **Direct execution** - No copy-paste, immediate results
3. **Learning loop** - ADAM remembers successful queries
4. **Context aware** - Understands your schema and data

### Integration with ADAM's Memory
- Successful queries stored in memory
- Query patterns learned over time
- Performance metrics tracked
- Automatic optimization suggestions

## Future Enhancements

### Planned Features
1. **Query optimization advisor** - Suggest improvements
2. **Automatic indexing recommendations**
3. **Cost estimation before execution**
4. **Query result caching**
5. **Automated report generation**

### Potential Integrations
- dbt Cloud API
- Snowflake query history analysis
- Automated data profiling
- Schema change detection

## Testing

Run tests:
```bash
# Simple test (no credentials needed)
python test_snowflake_simple.py

# Full test (requires Snowflake credentials)
python test_snowflake_tools.py
```

## Summary

ADAM can now:
✅ Generate production-ready Snowflake queries
✅ Execute queries and return results
✅ Analyze data quality automatically
✅ Create dbt models and stored procedures
✅ Learn from query patterns to improve over time

This makes ADAM a powerful data analyst that can:
- Write SQL faster than humans
- Execute and validate results
- Learn from successful patterns
- Optimize based on Snowflake best practices

---

*With this integration, ADAM becomes your AI-powered data analyst, capable of writing, executing, and learning from SQL queries.*