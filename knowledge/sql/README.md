# Snowflake SQL Knowledge Integration for ADAM

## Overview
ADAM now has comprehensive Snowflake SQL optimization knowledge to write high-performance, properly formatted SQL queries following your organization's best practices.

## What We've Implemented

### 1. Knowledge Base Files
- **`SNOWFLAKE_SQL_KNOWLEDGE.md`** (13KB+) - Comprehensive SQL best practices guide
- **`snowflake_sql_patterns.yaml`** (15KB+) - Structured patterns and templates

### 2. SQL Knowledge Service
- **Location**: `/src/adam_v2/services/sql_knowledge_service.py`
- **Features**:
  - Intelligent SQL query detection
  - Query optimization analysis
  - Automatic UPPERCASE keyword formatting
  - Anti-pattern detection and fixes
  - Snowflake-specific optimizations
  - Performance suggestions

### 3. Integration Features
- Automatic detection of SQL-related queries
- Enhancement with optimization tips
- SQL keyword uppercase enforcement
- Query pattern templates
- Self-learning from successful queries

## Core Principles Enforced

### 1. **UPPERCASE SQL Keywords**
All SQL keywords are automatically converted to UPPERCASE:
```sql
-- ADAM will write:
SELECT 
    user_id,
    SUM(amount) AS total_amount
FROM transactions
WHERE created_at >= '2024-01-01'
```

### 2. **Performance First**
- Never use `SELECT *` in production
- Filter early for partition pruning
- Join on indexed/clustered columns
- Avoid functions on filtered columns

### 3. **Proper Formatting**
- One clause per line
- 4-space indentation
- Meaningful aliases
- Comments for complex logic

## How to Use

### 1. Start the Backend
```bash
# Restart to load SQL knowledge
PYTHONPATH=/Users/vitoryago/ADAM/src python -m adam_v2.main
```

### 2. Example Queries for ADAM

#### Optimize a Query:
```
"Optimize this SQL query:
select * from orders o, customers c 
where o.cust_id = c.id and date(o.created_at) = '2024-01-01'"
```

ADAM will:
- Convert to UPPERCASE keywords
- Replace `SELECT *` with specific columns
- Fix implicit join to explicit JOIN
- Remove function from date filter
- Add proper formatting

#### Write High-Performance SQL:
```
"Write a SQL query to find top 10 customers by revenue in the last 30 days"
```

#### Get Optimization Tips:
```
"What are Snowflake SQL best practices for large table joins?"
```

#### Fix Anti-Patterns:
```
"Review this query for performance issues:
SELECT DISTINCT * FROM large_table ORDER BY created_at"
```

## SQL Optimization Features

### 1. Anti-Pattern Detection
ADAM detects and fixes:
- `SELECT *` usage
- Functions on filtered columns
- Missing WHERE clauses
- Implicit cross joins
- OR conditions in JOINs
- Unnecessary DISTINCT/ORDER BY
- Lowercase SQL keywords

### 2. Snowflake-Specific Optimizations
- **Clustering alignment**: Queries align with table clustering keys
- **Partition pruning**: Direct column comparisons for micro-partition pruning
- **Result caching**: Identical queries for cache hits
- **Warehouse sizing**: Recommendations based on query complexity
- **Time Travel settings**: TRANSIENT for staging tables

### 3. Query Patterns Available
- Basic SELECT with filters
- Multi-table JOINs
- Window functions (ROW_NUMBER, RANK, etc.)
- CTEs (Common Table Expressions)
- Aggregations with GROUP BY
- Incremental MERGE patterns
- Data quality checks

## Example Transformations

### Before (Poor Performance):
```sql
select * from orders 
where year(order_date) = 2024 
order by order_date
```

### After (ADAM Optimized):
```sql
SELECT 
    order_id,
    customer_id,
    order_amount,
    order_date
FROM orders
WHERE order_date >= '2024-01-01' 
    AND order_date < '2025-01-01'
ORDER BY order_date
LIMIT 1000
```

## Query Analysis Features

When you provide SQL to ADAM, it will:

1. **Analyze for Issues**:
   - Scan efficiency
   - Join optimization
   - Filter effectiveness
   - Sort necessity

2. **Provide Suggestions**:
   - Specific improvements with examples
   - Severity levels (critical/warning/info)
   - Performance impact estimates

3. **Format Properly**:
   - UPPERCASE all keywords
   - Proper indentation
   - Clear structure

## Integration with DBT

When writing SQL for DBT models, ADAM:
- Applies same optimization principles
- Uses DBT-specific syntax (ref(), source())
- Suggests incremental patterns for large datasets
- Maintains consistency between raw SQL and DBT SQL

## Monitoring Queries

ADAM can help write monitoring queries:

```sql
-- Find slow queries
SELECT 
    query_text,
    total_elapsed_time / 1000 AS seconds,
    bytes_scanned / POWER(1024, 3) AS gb_scanned
FROM TABLE(INFORMATION_SCHEMA.QUERY_HISTORY())
WHERE total_elapsed_time > 30000
ORDER BY total_elapsed_time DESC
LIMIT 20
```

## Self-Learning

ADAM stores successful query patterns and learns from:
- Execution time
- Bytes scanned
- Rows returned
- Efficiency scores

This helps ADAM suggest similar optimized patterns for new queries.

## Testing SQL Knowledge

Ask ADAM these questions to test:

1. **Optimization**: "How can I optimize a query that scans 1TB of data?"
2. **Best Practices**: "What are Snowflake SQL formatting standards?"
3. **Anti-Patterns**: "Why should I avoid SELECT * in production?"
4. **Performance**: "How do clustering keys improve query performance?"
5. **Specific SQL**: "Write an efficient query to deduplicate data"

## Benefits

✅ **Performance**: Queries run faster with proper optimization  
✅ **Consistency**: All SQL follows same standards (UPPERCASE keywords)  
✅ **Cost Savings**: Less data scanned = lower Snowflake costs  
✅ **Maintainability**: Well-formatted, documented queries  
✅ **Learning**: ADAM improves based on query patterns  

## Architecture

```
User Query → LLM Service → SQL Detection
                ↓
         SQL Knowledge Service
                ↓
    ┌──────────────────────────────┐
    │  SNOWFLAKE_SQL_KNOWLEDGE.md   │
    │  snowflake_sql_patterns.yaml  │
    │  Memory (learned patterns)     │
    └──────────────────────────────┘
                ↓
         Optimized SQL Response
```

## Troubleshooting

If SQL optimization isn't working:
1. Check knowledge files exist: `ls SNOWFLAKE_SQL_*.md snowflake_sql_patterns.yaml`
2. Restart backend to reload knowledge
3. Check logs for "SQL Knowledge Service initialized"
4. Verify query contains SQL keywords

---

*Snowflake SQL Knowledge Integration completed! ADAM now writes performant, properly formatted SQL with UPPERCASE keywords.* 🚀