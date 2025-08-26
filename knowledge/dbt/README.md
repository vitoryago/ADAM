# DBT Knowledge Base for ADAM

## Overview
ADAM has comprehensive DBT (Data Build Tool) knowledge to help you build high-performance, maintainable data transformation pipelines following best practices.

## What's Included

### 1. Knowledge Base Files
- **`DBT_KNOWLEDGE.md`** (12.8KB) - Comprehensive DBT best practices guide
- **`dbt_patterns.yaml`** (15.2KB) - Structured patterns and templates

### 2. DBT Knowledge Service
- **Location**: `/src/adam_v2/services/dbt_knowledge_service.py`
- **Features**:
  - Intelligent DBT context detection
  - Model layer identification
  - Template generation
  - Pattern suggestion
  - Convention validation
  - Incremental strategy selection

### 3. Integration Features
- Automatic detection of DBT-related queries
- Enhancement with best practices
- Model template generation
- Anti-pattern detection
- Self-learning from successful models

## Core Principles

### 1. **Layer Architecture**
Clear separation of concerns:
```
Raw Data → Staging → Intermediate → Marts
```

### 2. **Naming Conventions**
```yaml
staging:      stg_<source>__<entity>
intermediate: int_<entity>__<verb>
facts:        fct_<entity>
dimensions:   dim_<entity>
```

### 3. **Materialization Strategy**
- **Staging**: Views (light transforms only)
- **Intermediate**: Ephemeral (complex logic)
- **Marts**: Tables or Incremental (business-ready)

### 4. **Snowflake Optimizations**
- Cluster keys aligned with query patterns
- Transient tables for staging
- Auto-clustering for large fact tables
- Appropriate warehouse sizing

## How to Use

### 1. Start the Backend
```bash
# ADAM will automatically load DBT knowledge
PYTHONPATH=/Users/vitoryago/ADAM/src python -m adam_v2.main
```

### 2. Example Queries for ADAM

#### Create a Model:
```
"Create a staging model for the orders table from the raw schema"
```

ADAM will:
- Generate proper naming (stg_raw__orders)
- Apply staging conventions
- Include only light transformations
- Add appropriate config

#### Convert Looker PDT:
```
"Convert this Looker derived table to a DBT model:
[paste PDT SQL]"
```

#### Optimize Incremental:
```
"What's the best incremental strategy for a fact table with updates?"
```

#### Fix Anti-Patterns:
```
"Review this DBT model for best practices:
[paste model SQL]"
```

## DBT Features Covered

### 1. Model Types
- Source definitions
- Staging models
- Intermediate transformations
- Fact tables
- Dimension tables
- Data marts

### 2. Configurations
- Materializations (view, table, incremental, ephemeral)
- Clustering keys
- Partitioning
- Tags and meta
- Pre/post hooks

### 3. Testing
- Schema tests (unique, not_null, relationships)
- Custom data tests
- Source freshness
- Volume anomaly detection

### 4. Documentation
- Model descriptions
- Column descriptions
- Business logic documentation
- Lineage tracking

### 5. Macros & Utilities
- Custom macros
- Incremental helpers
- Date spine generation
- Surrogate key generation

## Example Model Generation

When you ask ADAM to create a DBT model, it generates:

```sql
-- Staging model for orders from raw source
-- Grain: One row per order
-- Frequency: Refreshed daily

{{
    config(
        materialized='view',
        cluster_by=['order_date', 'customer_id'],
        tags=['staging', 'daily'],
        transient=true
    )
}}

WITH source AS (
    SELECT * FROM {{ source('raw', 'orders') }}
),

renamed AS (
    SELECT
        -- Primary Key
        id AS order_id,
        
        -- Foreign Keys
        customer_id,
        product_id,
        
        -- Dimensions
        status AS order_status,
        channel AS order_channel,
        
        -- Measures
        amount AS order_amount,
        quantity AS order_quantity,
        
        -- Timestamps
        created_at,
        updated_at,
        
        -- Metadata
        CURRENT_TIMESTAMP() AS _loaded_at
    FROM source
    WHERE NOT is_deleted  -- Filter soft deletes
)

SELECT * FROM renamed
```

## Incremental Patterns

ADAM knows multiple incremental strategies:

### 1. Append-Only (Events)
```sql
{{ config(
    materialized='incremental',
    unique_key='event_id',
    on_schema_change='fail'
) }}

SELECT * FROM source
{% if is_incremental() %}
    WHERE created_at > (SELECT MAX(created_at) FROM {{ this }})
{% endif %}
```

### 2. Merge with Updates
```sql
{{ config(
    materialized='incremental',
    unique_key='id',
    merge_update_columns=['status', 'amount', 'updated_at']
) }}

SELECT * FROM source
{% if is_incremental() %}
    WHERE updated_at >= DATEADD('day', -3, CURRENT_DATE())
{% endif %}
```

## Self-Learning

ADAM learns from successful DBT patterns:
- Stores models that pass tests
- Tracks execution performance
- Suggests similar patterns for new models
- Improves recommendations over time

## Testing Patterns

ADAM suggests comprehensive testing:

```yaml
models:
  - name: fct_orders
    columns:
      - name: order_id
        tests:
          - unique
          - not_null
      - name: customer_id
        tests:
          - relationships:
              to: ref('dim_customers')
              field: customer_id
      - name: order_amount
        tests:
          - not_null
          - positive_value
    tests:
      - volume_anomaly:
          lookback_days: 30
          threshold: 0.5
```

## Benefits

✅ **Consistency**: All models follow same standards  
✅ **Performance**: Optimized for Snowflake  
✅ **Maintainability**: Clear structure and naming  
✅ **Quality**: Comprehensive testing coverage  
✅ **Documentation**: Self-documenting patterns  

## Troubleshooting

If DBT knowledge isn't working:
1. Check knowledge files exist: `ls knowledge/dbt/`
2. Restart backend to reload knowledge
3. Check logs for "DBT Knowledge Service initialized"
4. Verify query contains DBT-related context

---

*Building better data pipelines with intelligent DBT assistance!* 🚀