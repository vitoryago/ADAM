# Looker PDT to DBT Migration Knowledge Base

## Overview
Comprehensive guide for migrating Looker Persistent Derived Tables (PDTs) to DBT models while maintaining backward compatibility with Looker views.

## Migration Philosophy

### Core Principles
1. **One PDT = One DBT Model**: Each PDT becomes a dedicated DBT model
2. **Preserve Business Logic**: Maintain exact SQL logic during initial migration
3. **Optimize Later**: First migrate, then optimize for DBT best practices
4. **Backward Compatibility**: Generate Looker views that reference DBT models
5. **Incremental Where Possible**: Convert PDTs to incremental models for better performance

## PDT to DBT Conversion Patterns

### 1. Basic PDT to DBT Model

**Looker PDT**:
```lookml
view: user_facts {
  derived_table: {
    sql: 
      SELECT 
        user_id,
        COUNT(*) as lifetime_orders,
        SUM(amount) as lifetime_value,
        MIN(created_at) as first_order_date,
        MAX(created_at) as last_order_date
      FROM orders
      GROUP BY user_id
    ;;
    persist_for: "24 hours"
    indexes: ["user_id"]
  }
}
```

**DBT Equivalent**:
```sql
-- models/marts/core/fct_user_lifetime_metrics.sql
{{
    config(
        materialized='table',
        cluster_by=['user_id'],
        tags=['looker_migration', 'user_facts'],
        meta={
            'looker_view': 'user_facts',
            'original_persist': '24 hours'
        }
    )
}}

SELECT 
    user_id,
    COUNT(*) AS lifetime_orders,
    SUM(amount) AS lifetime_value,
    MIN(created_at) AS first_order_date,
    MAX(created_at) AS last_order_date
FROM {{ ref('stg_orders') }}
GROUP BY user_id
```

**Generated Looker View**:
```lookml
# This view references the DBT model instead of using PDT
view: user_facts {
  sql_table_name: analytics.fct_user_lifetime_metrics ;;
  
  dimension: user_id {
    type: number
    primary_key: yes
    sql: ${TABLE}.user_id ;;
  }
  
  measure: lifetime_orders {
    type: number
    sql: ${TABLE}.lifetime_orders ;;
  }
  
  measure: lifetime_value {
    type: number
    value_format_name: usd
    sql: ${TABLE}.lifetime_value ;;
  }
}
```

### 2. PDT with Datagroup to Incremental Model

**Looker PDT with Datagroup**:
```lookml
datagroup: daily_etl {
  sql_trigger: SELECT MAX(updated_at) FROM etl_log ;;
  max_cache_age: "1 hour"
}

view: daily_revenue {
  derived_table: {
    datagroup_trigger: daily_etl
    partition_keys: ["date"]
    sql:
      SELECT 
        DATE_TRUNC('day', created_at) as date,
        product_id,
        SUM(revenue) as daily_revenue,
        COUNT(DISTINCT user_id) as unique_users
      FROM events
      WHERE created_at >= '2020-01-01'
      GROUP BY 1, 2
    ;;
  }
}
```

**DBT Incremental Model**:
```sql
-- models/marts/revenue/fct_daily_product_revenue.sql
{{
    config(
        materialized='incremental',
        unique_key=['date', 'product_id'],
        on_schema_change='sync_all_columns',
        cluster_by=['date', 'product_id'],
        incremental_strategy='merge',
        tags=['looker_migration', 'daily_revenue']
    )
}}

WITH daily_metrics AS (
    SELECT 
        DATE_TRUNC('day', created_at) AS date,
        product_id,
        SUM(revenue) AS daily_revenue,
        COUNT(DISTINCT user_id) AS unique_users,
        MAX(created_at) AS max_event_time  -- For incremental filter
    FROM {{ ref('stg_events') }}
    
    {% if is_incremental() %}
        -- Look back 3 days for late-arriving data
        WHERE created_at >= (
            SELECT DATEADD('day', -3, MAX(date)) 
            FROM {{ this }}
        )
    {% else %}
        WHERE created_at >= '2020-01-01'
    {% endif %}
    
    GROUP BY 1, 2
)

SELECT 
    date,
    product_id,
    daily_revenue,
    unique_users,
    CURRENT_TIMESTAMP() AS dbt_updated_at
FROM daily_metrics
```

### 3. Complex PDT with Multiple CTEs

**Looker Complex PDT**:
```lookml
view: customer_cohorts {
  derived_table: {
    sql:
      WITH first_purchase AS (
        SELECT 
          user_id,
          MIN(DATE_TRUNC('month', created_at)) as cohort_month
        FROM orders
        GROUP BY user_id
      ),
      monthly_revenue AS (
        SELECT 
          user_id,
          DATE_TRUNC('month', created_at) as revenue_month,
          SUM(amount) as monthly_amount
        FROM orders
        GROUP BY 1, 2
      )
      SELECT 
        fp.cohort_month,
        mr.revenue_month,
        DATEDIFF('month', fp.cohort_month, mr.revenue_month) as months_since_first,
        COUNT(DISTINCT fp.user_id) as cohort_users,
        SUM(mr.monthly_amount) as cohort_revenue
      FROM first_purchase fp
      JOIN monthly_revenue mr ON fp.user_id = mr.user_id
      GROUP BY 1, 2, 3
    ;;
    persist_for: "12 hours"
  }
}
```

**DBT Model with Proper Layering**:
```sql
-- models/intermediate/revenue/int_user_cohort_base.sql
{{
    config(
        materialized='ephemeral'
    )
}}

SELECT 
    user_id,
    MIN(DATE_TRUNC('month', created_at)) AS cohort_month
FROM {{ ref('stg_orders') }}
GROUP BY user_id

-- models/intermediate/revenue/int_user_monthly_revenue.sql
{{
    config(
        materialized='ephemeral'
    )
}}

SELECT 
    user_id,
    DATE_TRUNC('month', created_at) AS revenue_month,
    SUM(amount) AS monthly_amount
FROM {{ ref('stg_orders') }}
GROUP BY 1, 2

-- models/marts/analytics/fct_customer_cohorts.sql
{{
    config(
        materialized='table',
        cluster_by=['cohort_month', 'revenue_month'],
        tags=['looker_migration', 'customer_cohorts']
    )
}}

WITH first_purchase AS (
    SELECT * FROM {{ ref('int_user_cohort_base') }}
),

monthly_revenue AS (
    SELECT * FROM {{ ref('int_user_monthly_revenue') }}
)

SELECT 
    fp.cohort_month,
    mr.revenue_month,
    DATEDIFF('month', fp.cohort_month, mr.revenue_month) AS months_since_first,
    COUNT(DISTINCT fp.user_id) AS cohort_users,
    SUM(mr.monthly_amount) AS cohort_revenue,
    AVG(mr.monthly_amount) AS avg_user_revenue,
    CURRENT_TIMESTAMP() AS dbt_updated_at
FROM first_purchase fp
INNER JOIN monthly_revenue mr 
    ON fp.user_id = mr.user_id
GROUP BY 1, 2, 3
```

## Looker-Specific Features and DBT Equivalents

### Persist Strategies

| Looker PDT | DBT Equivalent | Notes |
|------------|----------------|-------|
| `persist_for: "24 hours"` | `materialized='table'` + scheduled refresh | Use DBT Cloud or Airflow for scheduling |
| `datagroup_trigger` | `materialized='incremental'` | More efficient than full refresh |
| `sql_trigger_value` | DBT `on-run-start` hook | Check condition before running |
| `partition_keys` | `cluster_by` config | Snowflake clustering for performance |
| `indexes` | `cluster_by` or `post-hook` | Platform-specific indexing |

### Distribution and Sorting

**Looker**:
```lookml
derived_table: {
  distribution_style: "even"
  sortkeys: ["created_at"]
  indexes: ["user_id", "product_id"]
}
```

**DBT**:
```sql
{{
    config(
        materialized='table',
        dist='even',  -- Redshift
        sort='created_at',  -- Redshift
        cluster_by=['user_id', 'product_id']  -- Snowflake/BigQuery
    )
}}
```

## Migration Process

### Step 1: Analyze PDT
1. Identify PDT type (static, scheduled, triggered)
2. Extract SQL logic
3. Identify dependencies
4. Note performance optimizations (indexes, partitions)
5. Document refresh schedule

### Step 2: Design DBT Model
1. Choose appropriate layer (staging, intermediate, marts)
2. Select materialization strategy
3. Plan incremental logic if applicable
4. Design test coverage
5. Document business logic

### Step 3: Create DBT Model
1. Write SQL with DBT best practices
2. Add appropriate config
3. Use refs for dependencies
4. Add clustering/partitioning
5. Include metadata for tracking

### Step 4: Generate Looker View
1. Create view referencing DBT model
2. Map all dimensions and measures
3. Preserve Looker-specific features
4. Maintain field descriptions
5. Update explores to use new view

### Step 5: Testing and Validation
1. Compare row counts
2. Validate key metrics
3. Check performance
4. Test incremental logic
5. Verify Looker dashboards

## Real-World PDT Patterns from Your Codebase

### Pattern 1: UNION ALL Event Streams (Segment Events)

**Your PDT Example: mfeed_pages_view**
```lookml
derived_table: {
  sql:
    SELECT columns FROM segment_cnf_content_feed_prod.pages
    UNION ALL
    SELECT columns FROM segment_cnf_content_feed_prod.content_feed_vertical_view_rendered
    UNION ALL
    SELECT columns FROM segment_cnf_content_feed_prod.content_feed_horizontal_view_rendered
}
```

**DBT Migration Approach**:
```sql
-- models/intermediate/segment/int_content_feed_events_union.sql
{{
    config(
        materialized='incremental',
        unique_key='id',
        on_schema_change='sync_all_columns',
        cluster_by=['timestamp', 'anonymous_id'],
        incremental_strategy='merge'
    )
}}

WITH pages_events AS (
    SELECT
        id,
        anonymous_id,
        channel,
        zone,
        device_id,
        referrer,
        search,
        NULL::TEXT AS event_text,
        CASE 
            WHEN search = 'scroll=horizontal' THEN 'Preview' 
            ELSE 'Vertical Feed' 
        END AS page_render_event_type,
        timestamp
    FROM {{ source('segment_cnf_content_feed_prod', 'pages') }}
    {% if is_incremental() %}
        WHERE timestamp > (SELECT MAX(timestamp) FROM {{ this }})
    {% endif %}
),

vertical_view_events AS (
    SELECT
        id,
        anonymous_id,
        channel,
        zone,
        device_id,
        referrer,
        NULL::TEXT AS search,
        event_text,
        CASE 
            WHEN event_text = 'content_feed_vertical_view_rendered' THEN 'Vertical Feed' 
            ELSE 'Preview' 
        END AS page_render_event_type,
        timestamp
    FROM {{ source('segment_cnf_content_feed_prod', 'content_feed_vertical_view_rendered') }}
    {% if is_incremental() %}
        WHERE timestamp > (SELECT MAX(timestamp) FROM {{ this }})
    {% endif %}
),

horizontal_view_events AS (
    SELECT
        id,
        anonymous_id,
        channel,
        zone,
        device_id,
        referrer,
        NULL::TEXT AS search,
        event_text,
        CASE 
            WHEN event_text = 'content_feed_horizontal_view_rendered' THEN 'Preview' 
            ELSE 'Vertical Feed' 
        END AS page_render_event_type,
        timestamp
    FROM {{ source('segment_cnf_content_feed_prod', 'content_feed_horizontal_view_rendered') }}
    {% if is_incremental() %}
        WHERE timestamp > (SELECT MAX(timestamp) FROM {{ this }})
    {% endif %}
)

SELECT * FROM pages_events
UNION ALL
SELECT * FROM vertical_view_events
UNION ALL
SELECT * FROM horizontal_view_events
```

**Key Optimization**: Convert to incremental since Segment events are append-only!

### Pattern 2: First/Last Value Aggregations (Account First Monetization)

**Your PDT Example: first_monetized_demand_booked**
```lookml
derived_table: {
  sql: 
    SELECT account_name, account_id, MIN(monetized_booked_revenue) AS first_monetized_booked_revenue
    FROM (nested query with joins)
    GROUP BY account_name, account_id
}
```

**DBT Migration Approach**:
```sql
-- models/intermediate/finance/int_account_revenue_events.sql
{{
    config(
        materialized='ephemeral'
    )
}}

SELECT 
    a.name AS account_name,
    a.id AS account_id,
    CASE 
        WHEN bbh.revenue > 0 THEN DATE_TRUNC('day', bbh.booked_at)
    END AS monetized_booked_revenue,
    bbh.booked_at,
    bbh.revenue
FROM {{ ref('stg_booked_supply_by_demand_by_hour_financials') }} AS bbh
INNER JOIN {{ ref('stg_sub_accounts') }} AS demand_sa
    ON bbh.demand_sub_account_id = demand_sa.id
INNER JOIN {{ ref('stg_accounts') }} AS a
    ON demand_sa.account_id = a.id

-- models/marts/finance/fct_first_monetized_demand.sql
{{
    config(
        materialized='table',
        cluster_by=['account_id'],
        tags=['looker_migration', 'first_monetization']
    )
}}

SELECT 
    account_name,
    account_id,
    MIN(monetized_booked_revenue) AS first_monetized_booked_revenue,
    MAX(monetized_booked_revenue) AS last_monetized_booked_revenue,
    COUNT(DISTINCT monetized_booked_revenue) AS monetized_days_count
FROM {{ ref('int_account_revenue_events') }}
WHERE monetized_booked_revenue IS NOT NULL
GROUP BY 
    account_name,
    account_id
```

### Pattern 3: Complex CTEs with Window Functions (Status Tracking)

**Your PDT Example: prove_prefill**
```lookml
derived_table: {
  sql: 
    WITH latest_prefill_status AS (
        SELECT * FROM (
            SELECT *, ROW_NUMBER() OVER (PARTITION BY source_uuid ORDER BY created_at DESC) AS rn
            FROM public.prove_prefill
        ) sub
        WHERE rn = 1
    )
    SELECT lead_uuid, aggregations...
}
```

**DBT Migration Approach**:
```sql
-- models/intermediate/prove/int_prove_prefill_latest.sql
{{
    config(
        materialized='view'  -- or table if performance requires
    )
}}

SELECT *
FROM (
    SELECT
        *,
        ROW_NUMBER() OVER (
            PARTITION BY source_uuid 
            ORDER BY created_at DESC
        ) AS rn
    FROM {{ ref('stg_prove_prefill') }}
) ranked
WHERE rn = 1

-- models/marts/prove/fct_prove_prefill_summary.sql
{{
    config(
        materialized='incremental',
        unique_key='lead_uuid',
        on_schema_change='sync_all_columns'
    )
}}

WITH latest_status AS (
    SELECT * FROM {{ ref('int_prove_prefill_latest') }}
    {% if is_incremental() %}
        WHERE created_at >= (SELECT MAX(created_at) FROM {{ this }})
    {% endif %}
),

step_aggregations AS (
    SELECT
        pps.prefill_uuid,
        SUM(CASE WHEN pps.step_type = 'promptDobPhone' THEN 1 ELSE 0 END) AS prompt_dob_phone,
        SUM(CASE WHEN pps.step_type = 'initSmsOtp' THEN 1 ELSE 0 END) AS init_sms_otp,
        SUM(CASE WHEN pps.step_type = 'confirmSmsOtp' THEN 1 ELSE 0 END) AS confirm_sms_otp,
        -- ... other step aggregations
    FROM {{ ref('stg_prove_prefill_steps') }} pps
    GROUP BY pps.prefill_uuid
)

SELECT
    latest.source_uuid AS lead_uuid,
    COUNT(DISTINCT CASE WHEN latest.status = 'success' THEN latest.uuid END) AS prefill_success,
    COUNT(DISTINCT CASE WHEN latest.status = 'failure' THEN latest.uuid END) AS prefill_failure,
    COUNT(DISTINCT CASE WHEN latest.status = 'pending' THEN latest.uuid END) AS prefill_pending,
    steps.prompt_dob_phone,
    steps.init_sms_otp,
    steps.confirm_sms_otp,
    -- ... other fields
    CURRENT_TIMESTAMP() AS dbt_updated_at
FROM latest_status latest
LEFT JOIN step_aggregations steps
    ON steps.prefill_uuid = latest.uuid
GROUP BY 
    lead_uuid,
    steps.prompt_dob_phone,
    steps.init_sms_otp,
    steps.confirm_sms_otp
```

## Common Migration Patterns

### 1. Aggregate Tables
**Pattern**: PDTs that pre-aggregate data
**DBT Strategy**: Fact tables with appropriate grain
**Optimization**: Use incremental with lookback window

### 2. Denormalized Tables
**Pattern**: PDTs that join multiple sources
**DBT Strategy**: Intermediate models for joins, marts for final
**Optimization**: Leverage DBT's DAG for clarity

### 3. Event Stream Processing
**Pattern**: PDTs processing event data
**DBT Strategy**: Incremental models with event_time filter
**Optimization**: Use insert_overwrite for partitioned tables

### 4. Slowly Changing Dimensions
**Pattern**: PDTs tracking dimension changes
**DBT Strategy**: Snapshots or SCD Type 2 models
**Optimization**: Use DBT snapshots feature

### 5. Sessionization
**Pattern**: PDTs creating user sessions
**DBT Strategy**: Window functions in incremental model
**Optimization**: Process only new events incrementally

## Looker View Generation Templates

### Basic Dimension/Measure View
```lookml
view: {dbt_model_name} {
  sql_table_name: {database}.{schema}.{dbt_model_name} ;;
  label: "{business_friendly_name}"
  
  # Dimensions
  dimension: {field_name} {
    type: {looker_type}
    sql: ${TABLE}.{column_name} ;;
    description: "{field_description}"
  }
  
  # Measures
  measure: {measure_name} {
    type: {aggregate_type}
    sql: ${TABLE}.{column_name} ;;
    value_format_name: {format}
  }
}
```

### View with Drill Fields
```lookml
view: {dbt_model_name} {
  sql_table_name: {database}.{schema}.{dbt_model_name} ;;
  
  measure: total_revenue {
    type: sum
    sql: ${TABLE}.revenue ;;
    drill_fields: [detail*]
  }
  
  set: detail {
    fields: [
      user_id,
      created_at,
      product_name,
      revenue
    ]
  }
}
```

## Performance Optimization During Migration

### 1. Identify Optimization Opportunities
- Replace `SELECT *` with explicit columns
- Add clustering keys based on filter patterns
- Convert eligible PDTs to incremental models
- Decompose complex PDTs into layers

### 2. Incremental Conversion Checklist
- [ ] Unique key identified
- [ ] Timestamp column available
- [ ] Lookback window defined
- [ ] Late-arriving data handled
- [ ] Merge strategy selected

### 3. Testing Strategy
```yaml
# schema.yml for migrated model
models:
  - name: fct_migrated_pdt
    description: "Migrated from Looker PDT: {original_pdt_name}"
    meta:
      migrated_from: "looker_pdt"
      migration_date: "2024-12-10"
      original_persist: "24 hours"
    
    columns:
      - name: primary_key
        tests:
          - unique
          - not_null
      
      - name: amount
        tests:
          - not_null
          - positive_value
    
    tests:
      # Compare with original PDT
      - row_count_matches:
          compare_model: "looker_pdt_export"
      - metrics_match:
          compare_model: "looker_pdt_export"
          metrics: ["total_revenue", "user_count"]
```

## Migration Validation

### SQL to Compare Results
```sql
-- Compare row counts
WITH dbt_model AS (
    SELECT COUNT(*) as dbt_count 
    FROM analytics.fct_model
),
looker_pdt AS (
    SELECT COUNT(*) as pdt_count 
    FROM looker_scratch.LR_pdt_name
)
SELECT 
    dbt_count,
    pdt_count,
    dbt_count - pdt_count as difference,
    (dbt_count - pdt_count) / NULLIF(pdt_count, 0) * 100 as pct_difference
FROM dbt_model, looker_pdt;

-- Compare key metrics
WITH dbt_metrics AS (
    SELECT 
        SUM(revenue) as total_revenue,
        COUNT(DISTINCT user_id) as unique_users,
        AVG(order_value) as avg_order_value
    FROM analytics.fct_model
),
pdt_metrics AS (
    SELECT 
        SUM(revenue) as total_revenue,
        COUNT(DISTINCT user_id) as unique_users,
        AVG(order_value) as avg_order_value
    FROM looker_scratch.LR_pdt_name
)
SELECT 
    'Revenue' as metric,
    dbt.total_revenue as dbt_value,
    pdt.total_revenue as pdt_value,
    ABS(dbt.total_revenue - pdt.total_revenue) < 0.01 as matches
FROM dbt_metrics dbt, pdt_metrics pdt
```

## Automation Templates

### DBT Macro for Looker View Generation
```sql
-- macros/generate_looker_view.sql
{% macro generate_looker_view(model_name, schema_name='analytics') %}
  {% set columns = adapter.get_columns_in_relation(ref(model_name)) %}
  
view: {{ model_name }} {
  sql_table_name: {{ schema_name }}.{{ model_name }} ;;
  
  {% for column in columns %}
  dimension: {{ column.name }} {
    type: {% if column.data_type in ['INT', 'BIGINT', 'FLOAT'] %}number
          {% elif column.data_type in ['DATE'] %}date
          {% elif column.data_type in ['TIMESTAMP'] %}time
          {% else %}string{% endif %}
    sql: ${TABLE}.{{ column.name }} ;;
  }
  {% endfor %}
}
{% endmacro %}
```

## Best Practices

### Do's ✅
1. **Preserve Business Logic**: Keep the same logic initially, optimize later
2. **Document Everything**: Track what was migrated and why
3. **Test Thoroughly**: Validate data matches before switching
4. **Use DBT Features**: Leverage refs, tests, docs
5. **Plan Incrementally**: Convert to incremental where beneficial

### Don'ts ❌
1. **Don't Change Logic**: During migration, preserve exact behavior
2. **Don't Skip Testing**: Always validate against original PDT
3. **Don't Ignore Performance**: Check query times after migration
4. **Don't Break Dashboards**: Ensure Looker views maintain compatibility
5. **Don't Migrate All at Once**: Phase the migration

## Troubleshooting Common Issues

### Issue: Row Count Mismatch
**Causes**:
- Timezone differences
- Filter conditions
- Join logic changes

**Solution**:
```sql
-- Debug query to find differences
SELECT 'Only in DBT' as source, dbt.*
FROM analytics.fct_model dbt
LEFT JOIN looker_scratch.pdt pdt ON dbt.id = pdt.id
WHERE pdt.id IS NULL
UNION ALL
SELECT 'Only in PDT', pdt.*
FROM looker_scratch.pdt pdt
LEFT JOIN analytics.fct_model dbt ON pdt.id = dbt.id
WHERE dbt.id IS NULL
LIMIT 100;
```

### Issue: Performance Degradation
**Causes**:
- Missing indexes/clustering
- Full refresh vs incremental
- Suboptimal materialization

**Solution**:
- Add clustering keys
- Convert to incremental
- Review query plan

### Issue: Looker Dashboards Break
**Causes**:
- Column name changes
- Data type mismatches
- Missing aggregations

**Solution**:
- Maintain exact column names
- Cast data types explicitly
- Preserve measure definitions

---

*Seamless migration from Looker PDTs to DBT models with backward compatibility!* 🚀