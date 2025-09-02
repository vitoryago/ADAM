# Looker PDT to DBT Migration Knowledge Base

## Overview
This knowledge base enables ADAM to intelligently convert Looker Persistent Derived Tables (PDTs) to DBT models while maintaining backward compatibility with Looker.

## What ADAM Can Do

### 1. Convert PDTs to DBT Models
- Analyze PDT structure and logic
- Determine optimal DBT materialization strategy
- Generate production-ready DBT models
- Preserve business logic exactly
- Add performance optimizations

### 2. Generate Looker Views
- Create Looker views that reference DBT models
- Maintain all dimensions and measures
- Preserve field descriptions and formatting
- Ensure dashboard compatibility
- Generate explore configurations

### 3. Optimize During Migration
- Convert eligible PDTs to incremental models
- Add appropriate clustering/partitioning
- Split complex PDTs into layers
- Implement efficient refresh strategies
- Reduce compute costs

## Knowledge Files

- **`LOOKER_DBT_MIGRATION_KNOWLEDGE.md`** - Comprehensive migration guide with patterns and examples
- **`looker_dbt_patterns.yaml`** - Structured patterns for automated conversion
- **`README.md`** - This file, explaining the knowledge domain

## How to Use

### 1. Provide Your PDT
Share your Looker PDT code with ADAM:
```
"Convert this Looker PDT to DBT:
[paste your PDT code here]"
```

### 2. ADAM Will Provide

#### A. DBT Model
```sql
-- models/marts/[appropriate_name].sql
{{ config(...) }}

[Your optimized SQL]
```

#### B. Looker View
```lookml
view: your_view_name {
  sql_table_name: analytics.dbt_model ;;
  [dimensions and measures]
}
```

#### C. Migration Instructions
- Testing queries to validate data
- Performance optimization suggestions
- Integration steps for Looker

## Example Conversions

### Simple Aggregate PDT → DBT Table

**Input** (Looker PDT):
```lookml
view: daily_stats {
  derived_table: {
    sql: SELECT 
           DATE(created_at) as date,
           COUNT(*) as events,
           COUNT(DISTINCT user_id) as users
         FROM events
         GROUP BY 1 ;;
    persist_for: "24 hours"
  }
}
```

**Output** (DBT Model):
```sql
{{
    config(
        materialized='incremental',
        unique_key='date',
        cluster_by=['date']
    )
}}

SELECT 
    DATE(created_at) AS date,
    COUNT(*) AS events,
    COUNT(DISTINCT user_id) AS users,
    CURRENT_TIMESTAMP() AS dbt_updated_at
FROM {{ ref('stg_events') }}
{% if is_incremental() %}
    WHERE created_at >= (SELECT MAX(date) FROM {{ this }})
{% endif %}
GROUP BY 1
```

**Output** (Looker View):
```lookml
view: daily_stats {
  sql_table_name: analytics.fct_daily_stats ;;
  
  dimension_group: date {
    type: time
    timeframes: [raw, date, week, month]
    sql: ${TABLE}.date ;;
  }
  
  measure: total_events {
    type: sum
    sql: ${TABLE}.events ;;
  }
  
  measure: unique_users {
    type: sum
    sql: ${TABLE}.users ;;
  }
}
```

## Migration Patterns Covered

### PDT Types
- **Static PDTs** → DBT Tables
- **Scheduled PDTs** → DBT Tables with scheduled refresh
- **Triggered PDTs** → DBT Incremental models
- **Datagroup PDTs** → DBT Incremental with smart refresh

### Common Scenarios
1. **User Facts/Aggregates** - User-level rollups
2. **Daily/Monthly Rollups** - Time-based aggregations
3. **Cohort Analysis** - Complex cohort calculations
4. **Sessionization** - Event session creation
5. **Denormalization** - Pre-joined tables
6. **SCD Type 2** - Slowly changing dimensions

## Validation Process

ADAM provides validation queries:

```sql
-- Compare row counts
SELECT 
    (SELECT COUNT(*) FROM dbt_model) as dbt_rows,
    (SELECT COUNT(*) FROM looker_pdt) as pdt_rows;

-- Compare key metrics
SELECT 
    (SELECT SUM(revenue) FROM dbt_model) as dbt_revenue,
    (SELECT SUM(revenue) FROM looker_pdt) as pdt_revenue;
```

## Best Practices Applied

### ✅ DO's
- Preserve exact business logic initially
- Test thoroughly before switching
- Document the migration
- Use DBT best practices
- Optimize incrementally

### ❌ DON'T's
- Change logic during migration
- Skip validation
- Migrate everything at once
- Ignore performance
- Break existing dashboards

## Advanced Features

### 1. Incremental Conversion
ADAM can identify PDTs that are good candidates for incremental materialization:
- Event-based data
- Time-series data
- Large tables with timestamp columns
- Append-only patterns

### 2. Performance Optimization
ADAM suggests optimizations:
- Clustering keys based on query patterns
- Partitioning strategies
- Warehouse sizing
- Refresh scheduling

### 3. Testing Strategies
ADAM generates test configurations:
- Data quality tests
- Row count validations
- Metric comparisons
- Referential integrity

## Integration with Existing Knowledge

This migration knowledge works with:
- **DBT Knowledge** - For DBT best practices
- **SQL Knowledge** - For query optimization
- **Main LLM Service** - For intelligent detection

## Troubleshooting

### Common Issues and Solutions

1. **Row count mismatch**
   - Check timezone handling
   - Verify join conditions
   - Compare filter logic

2. **Performance degradation**
   - Add clustering keys
   - Convert to incremental
   - Review query plan

3. **Looker errors**
   - Check column names
   - Verify data types
   - Update explore configurations

## Request Examples

### Basic Migration
```
"Convert this Looker PDT to a DBT model:
view: user_facts {
  derived_table: {
    sql: SELECT user_id, COUNT(*) as orders
         FROM orders GROUP BY 1 ;;
  }
}"
```

### With Optimization
```
"Convert this PDT to an incremental DBT model with clustering:
[PDT code]"
```

### Full Migration
```
"Convert this PDT to DBT and generate the Looker view to reference it:
[PDT code]"
```

### Validation
```
"Generate validation queries to compare this PDT with its DBT replacement:
[PDT and DBT model names]"
```

## Adding Custom Patterns

To add new PDT patterns:

1. Edit `looker_dbt_patterns.yaml`
2. Add pattern under `common_patterns`
3. Include:
   - Typical SQL structure
   - DBT approach
   - Optimization strategy
   - Example conversion

## Performance Metrics

Typical improvements after migration:
- **Query Speed**: 20-50% faster with clustering
- **Refresh Time**: 60-80% faster with incremental
- **Storage**: 30-40% less with proper typing
- **Maintenance**: 50% less time with DBT testing

## Future Enhancements

Planned features:
- Automatic PDT discovery from LookML
- Bulk migration tools
- Performance comparison reports
- Automated testing generation
- Cost analysis pre/post migration

---

*Seamlessly migrate from Looker PDTs to DBT models with confidence!* 🚀