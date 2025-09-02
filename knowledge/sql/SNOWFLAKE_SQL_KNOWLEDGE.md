# Snowflake SQL Query Knowledge Base

> High-Performance SQL Query Writing for Snowflake Data Warehouse
> Version: Latest Snowflake | Focus: Performance, Readability, Best Practices

## Core Principles

1. **UPPERCASE SQL Keywords**: All SQL keywords (SELECT, FROM, WHERE, etc.) must be in UPPERCASE
2. **Select Only What You Need**: Never use SELECT * in production - explicitly list columns
3. **Filter Early, Filter Smart**: Apply WHERE clauses as early as possible for partition pruning
4. **Leverage Clustering**: Align queries with table clustering keys for optimal performance
5. **Format for Readability**: One clause per line, consistent indentation, meaningful aliases

## Performance Optimization Rules

### 1. Column Selection
```sql
-- ❌ BAD: Scans all columns
SELECT * FROM transactions

-- ✅ GOOD: Only needed columns
SELECT 
    transaction_id,
    user_id,
    amount,
    created_at
FROM transactions
```

### 2. Filter Optimization
```sql
-- ❌ BAD: Function on column prevents partition pruning
WHERE DATE(event_date) = '2024-01-01'
WHERE YEAR(created_at) = 2024

-- ✅ GOOD: Direct column comparison enables pruning
WHERE event_date = '2024-01-01'
WHERE created_at >= '2024-01-01' AND created_at < '2025-01-01'
```

### 3. Join Optimization
```sql
-- ❌ BAD: Join on expressions
FROM orders o
JOIN customers c
    ON UPPER(o.customer_code) = UPPER(c.code)

-- ✅ GOOD: Direct column joins
FROM orders o
JOIN customers c
    ON o.customer_id = c.id
```

### 4. Avoid Expensive Operations
```sql
-- ❌ BAD: Unnecessary DISTINCT on large dataset
SELECT DISTINCT * FROM large_table

-- ✅ GOOD: Use GROUP BY on specific columns
SELECT 
    user_id,
    date_id,
    MAX(amount) AS max_amount
FROM transactions
GROUP BY user_id, date_id

-- ❌ BAD: Cartesian product
FROM table1, table2  -- Missing join condition!

-- ✅ GOOD: Explicit join with condition
FROM table1 t1
INNER JOIN table2 t2
    ON t1.key = t2.key
```

## Snowflake-Specific Optimizations

### 1. Clustering Keys
```sql
-- Check clustering effectiveness
SELECT SYSTEM$CLUSTERING_INFORMATION('schema.table');

-- Query aligned with clustering
WHERE date_column >= '2024-01-01'  -- If clustered by date
    AND user_id = 12345            -- And user_id
```

### 2. Warehouse Sizing
```sql
-- For heavy queries, use appropriate warehouse
USE WAREHOUSE large_wh;

-- Or dynamically in query
ALTER SESSION SET WAREHOUSE = 'LARGE_WH';

-- Return to normal after
ALTER SESSION SET WAREHOUSE = 'MEDIUM_WH';
```

### 3. Result Caching
```sql
-- Enable result caching (same exact query returns instantly)
ALTER SESSION SET USE_CACHED_RESULT = TRUE;

-- Queries must be IDENTICAL (even spacing) to hit cache
SELECT user_id, amount FROM transactions WHERE date = '2024-01-01'
SELECT user_id, amount FROM transactions WHERE date = '2024-01-01'  -- Cache hit!
```

### 4. Micro-Partition Pruning
```sql
-- ✅ GOOD: Allows partition pruning
WHERE order_date >= DATEADD('day', -30, CURRENT_DATE())

-- ❌ BAD: Scans all partitions
WHERE DATEDIFF('day', order_date, CURRENT_DATE()) <= 30
```

## Query Patterns and Templates

### 1. Basic SELECT with Best Practices
```sql
SELECT 
    t.transaction_id,
    t.user_id,
    u.user_name,
    t.amount,
    t.created_at
FROM transactions AS t
INNER JOIN users AS u
    ON u.user_id = t.user_id
WHERE t.created_at >= '2024-01-01'
    AND t.status = 'COMPLETED'
    AND u.region = 'US'
ORDER BY t.created_at DESC
LIMIT 1000;
```

### 2. Aggregation with Window Functions
```sql
WITH daily_aggregates AS (
    SELECT 
        user_id,
        DATE_TRUNC('day', created_at) AS transaction_date,
        COUNT(*) AS transaction_count,
        SUM(amount) AS daily_amount,
        AVG(amount) AS avg_amount
    FROM transactions
    WHERE created_at >= DATEADD('month', -3, CURRENT_DATE())
    GROUP BY 
        user_id,
        DATE_TRUNC('day', created_at)
),
ranked_users AS (
    SELECT 
        user_id,
        transaction_date,
        daily_amount,
        ROW_NUMBER() OVER (
            PARTITION BY transaction_date 
            ORDER BY daily_amount DESC
        ) AS daily_rank
    FROM daily_aggregates
)
SELECT 
    user_id,
    transaction_date,
    daily_amount,
    daily_rank
FROM ranked_users
WHERE daily_rank <= 10
ORDER BY 
    transaction_date DESC,
    daily_rank;
```

### 3. Efficient Deduplication
```sql
-- Using ROW_NUMBER for deduplication (better than DISTINCT for complex cases)
WITH deduplicated AS (
    SELECT 
        *,
        ROW_NUMBER() OVER (
            PARTITION BY user_id, transaction_date
            ORDER BY created_at DESC
        ) AS rn
    FROM transactions
    WHERE created_at >= '2024-01-01'
)
SELECT 
    transaction_id,
    user_id,
    amount,
    created_at
FROM deduplicated
WHERE rn = 1;
```

### 4. Incremental Pattern
```sql
-- For incremental loads with lookback
MERGE INTO target_table AS tgt
USING (
    SELECT 
        user_id,
        metric_date,
        SUM(amount) AS daily_amount
    FROM source_table
    WHERE updated_at >= (
        SELECT DATEADD('day', -3, MAX(metric_date))
        FROM target_table
    )
    GROUP BY user_id, metric_date
) AS src
ON tgt.user_id = src.user_id
    AND tgt.metric_date = src.metric_date
WHEN MATCHED THEN
    UPDATE SET 
        daily_amount = src.daily_amount,
        updated_at = CURRENT_TIMESTAMP()
WHEN NOT MATCHED THEN
    INSERT (user_id, metric_date, daily_amount, updated_at)
    VALUES (src.user_id, src.metric_date, src.daily_amount, CURRENT_TIMESTAMP());
```

### 5. Semi-Join Pattern (EXISTS)
```sql
-- More efficient than IN for large datasets
SELECT 
    c.customer_id,
    c.customer_name,
    c.email
FROM customers AS c
WHERE EXISTS (
    SELECT 1
    FROM orders AS o
    WHERE o.customer_id = c.customer_id
        AND o.order_date >= '2024-01-01'
        AND o.status = 'SHIPPED'
);
```

## SQL Formatting Standards

### 1. Clause Structure
```sql
SELECT                          -- Main clauses start at column 0
    column1,                    -- Columns indented 4 spaces
    column2,
    SUM(amount) AS total       -- Meaningful aliases
FROM schema.table AS t         -- Table aliases
INNER JOIN other_table AS o    -- JOINs at same level as FROM
    ON o.id = t.other_id       -- ON conditions indented
    AND o.status = 'ACTIVE'    -- Additional conditions aligned
WHERE t.created_at >= '2024-01-01'
    AND t.region IN ('US', 'CA')  -- Logical operators at line start
GROUP BY 
    column1,
    column2
HAVING SUM(amount) > 1000
ORDER BY 
    total DESC,
    column1
LIMIT 100;
```

### 2. CTE Structure
```sql
WITH base_data AS (
    -- First CTE: Get base data
    SELECT 
        user_id,
        transaction_date,
        amount
    FROM transactions
    WHERE status = 'COMPLETED'
),
aggregated AS (
    -- Second CTE: Aggregate
    SELECT 
        user_id,
        COUNT(*) AS transaction_count,
        SUM(amount) AS total_amount
    FROM base_data
    GROUP BY user_id
),
final AS (
    -- Third CTE: Add calculations
    SELECT 
        user_id,
        transaction_count,
        total_amount,
        total_amount / NULLIF(transaction_count, 0) AS avg_amount
    FROM aggregated
)
-- Main query
SELECT * FROM final
WHERE transaction_count > 10;
```

### 3. Subquery Formatting
```sql
SELECT 
    c.customer_id,
    c.customer_name,
    (
        -- Subquery indented and commented
        SELECT COUNT(*)
        FROM orders
        WHERE customer_id = c.customer_id
            AND order_date >= '2024-01-01'
    ) AS recent_order_count
FROM customers AS c
WHERE c.status = 'ACTIVE';
```

## Naming Conventions

### 1. Object Naming
- **Tables**: Singular nouns, UPPERCASE or lowercase with underscores
  - `CUSTOMER` or `customer`
  - `ORDER_ITEM` or `order_item`
- **Columns**: Descriptive, include units/context
  - `amount_usd` not just `amount`
  - `created_at` for timestamps
  - `is_active` for booleans
  - `user_id` for foreign keys

### 2. Aliases
```sql
-- Meaningful table aliases
FROM customers AS c           -- Single letter for simple queries
FROM customer_orders AS co    -- Abbreviation for complex
FROM transaction_details AS txn_dtl  -- Clear abbreviation

-- Column aliases
SELECT 
    COUNT(*) AS total_count,   -- Not just 'count'
    SUM(amount) AS revenue_usd,  -- Include units
    AVG(score) AS avg_score     -- Clear meaning
```

### 3. CTEs and Temp Tables
```sql
-- Descriptive CTE names
WITH recent_active_users AS (...)
WITH daily_revenue_summary AS (...)

-- Temp table prefixes
CREATE TEMPORARY TABLE tmp_user_metrics AS ...
CREATE TRANSIENT TABLE stg_order_processing AS ...
```

## Anti-Patterns to Avoid

### 1. SELECT * 
```sql
-- ❌ NEVER in production
SELECT * FROM large_table

-- ✅ Always specify columns
SELECT col1, col2, col3 FROM large_table
```

### 2. Implicit Joins
```sql
-- ❌ Old-style implicit joins
FROM orders, customers
WHERE orders.customer_id = customers.id

-- ✅ Explicit JOIN syntax
FROM orders o
INNER JOIN customers c
    ON o.customer_id = c.id
```

### 3. OR in JOINs
```sql
-- ❌ OR conditions in joins prevent optimization
FROM orders o
JOIN customers c
    ON o.customer_id = c.id
    OR o.email = c.email

-- ✅ Use UNION for multiple join conditions
SELECT ... FROM orders o JOIN customers c ON o.customer_id = c.id
UNION
SELECT ... FROM orders o JOIN customers c ON o.email = c.email
```

### 4. Unnecessary Sorting
```sql
-- ❌ Sorting when not needed
SELECT * FROM transactions ORDER BY created_at  -- Into a temp table?

-- ✅ Only sort for final output
INSERT INTO summary_table
SELECT ... FROM transactions  -- No ORDER BY needed for INSERT
```

## Performance Monitoring

### 1. Query Profile Analysis
```sql
-- After running a query, check profile
-- Look for:
-- - TableScan operations (might need better filters)
-- - Large data shuffles (might need different join strategy)
-- - Spilling to disk (might need larger warehouse)
```

### 2. Query History
```sql
-- Find slow queries
SELECT 
    query_text,
    total_elapsed_time / 1000 AS seconds,
    bytes_scanned / POWER(1024, 3) AS gb_scanned,
    warehouse_name,
    warehouse_size
FROM SNOWFLAKE.ACCOUNT_USAGE.QUERY_HISTORY
WHERE start_time >= DATEADD('day', -7, CURRENT_TIMESTAMP())
    AND total_elapsed_time > 60000  -- Queries over 60 seconds
ORDER BY total_elapsed_time DESC
LIMIT 20;
```

### 3. Table Clustering Health
```sql
-- Check if reclustering needed
SELECT 
    table_name,
    PARSE_JSON(SYSTEM$CLUSTERING_INFORMATION(table_schema || '.' || table_name)) AS clustering_info
FROM information_schema.tables
WHERE table_schema = 'MY_SCHEMA'
    AND clustering_key IS NOT NULL;
```

## Utility Functions and Helpers

### 1. Safe Division
```sql
-- Prevent division by zero
SELECT 
    numerator / NULLIF(denominator, 0) AS safe_ratio,
    COALESCE(numerator / NULLIF(denominator, 0), 0) AS ratio_with_default
FROM data;
```

### 2. Date Range Helpers
```sql
-- Current month
WHERE date_column >= DATE_TRUNC('month', CURRENT_DATE())
    AND date_column < DATEADD('month', 1, DATE_TRUNC('month', CURRENT_DATE()))

-- Last 30 days
WHERE date_column >= DATEADD('day', -30, CURRENT_DATE())

-- Previous complete month
WHERE date_column >= DATEADD('month', -1, DATE_TRUNC('month', CURRENT_DATE()))
    AND date_column < DATE_TRUNC('month', CURRENT_DATE())
```

### 3. JSON Parsing
```sql
-- Safe JSON extraction
SELECT 
    TRY_PARSE_JSON(json_column):field1::STRING AS field1,
    TRY_PARSE_JSON(json_column):field2.subfield::NUMBER AS numeric_field
FROM json_table
WHERE TRY_PARSE_JSON(json_column) IS NOT NULL;
```

## Query Optimization Checklist

Before running a production query, verify:

✅ **Columns**: Only selecting needed columns (no SELECT *)
✅ **Filters**: WHERE clauses don't use functions on columns
✅ **Joins**: Using proper JOIN syntax with direct column comparisons
✅ **Clustering**: Query filters align with table clustering keys
✅ **CTEs**: Not referencing same expensive CTE multiple times
✅ **Sorting**: Only using ORDER BY when necessary
✅ **Limits**: Using LIMIT for testing on large datasets
✅ **Format**: Proper indentation and UPPERCASE keywords
✅ **Aliases**: Meaningful table and column aliases
✅ **Comments**: Complex logic is documented

## Integration with DBT

When writing SQL for DBT models:
1. Follow the same performance principles
2. Use DBT's `{{ ref() }}` and `{{ source() }}` for table references
3. Leverage DBT's incremental patterns for large datasets
4. Test queries in Snowflake first, then convert to DBT syntax

## Example: Complete Optimized Query

```sql
-- Calculate top products by revenue per region for last 30 days
WITH recent_orders AS (
    -- Get recent orders with early filtering
    SELECT 
        o.order_id,
        o.product_id,
        o.region_id,
        o.quantity,
        o.unit_price,
        o.order_date
    FROM orders AS o
    WHERE o.order_date >= DATEADD('day', -30, CURRENT_DATE())
        AND o.status = 'COMPLETED'  -- Filter early
        AND o.is_valid = TRUE
),
product_revenue AS (
    -- Calculate revenue per product per region
    SELECT 
        ro.region_id,
        ro.product_id,
        p.product_name,
        p.category,
        COUNT(DISTINCT ro.order_id) AS order_count,
        SUM(ro.quantity) AS total_quantity,
        SUM(ro.quantity * ro.unit_price) AS total_revenue
    FROM recent_orders AS ro
    INNER JOIN products AS p
        ON p.product_id = ro.product_id
        AND p.is_active = TRUE  -- Additional filter in JOIN
    GROUP BY 
        ro.region_id,
        ro.product_id,
        p.product_name,
        p.category
),
ranked_products AS (
    -- Rank products within each region
    SELECT 
        pr.*,
        ROW_NUMBER() OVER (
            PARTITION BY pr.region_id 
            ORDER BY pr.total_revenue DESC
        ) AS revenue_rank
    FROM product_revenue AS pr
)
-- Final output
SELECT 
    rp.region_id,
    r.region_name,
    rp.product_id,
    rp.product_name,
    rp.category,
    rp.order_count,
    rp.total_quantity,
    rp.total_revenue,
    rp.revenue_rank
FROM ranked_products AS rp
INNER JOIN regions AS r
    ON r.region_id = rp.region_id
WHERE rp.revenue_rank <= 10  -- Top 10 per region
ORDER BY 
    rp.region_id,
    rp.revenue_rank;
```

---
*This knowledge base ensures ADAM writes performant, readable Snowflake SQL with proper UPPERCASE keywords and optimal patterns.*