# DBT Knowledge Integration for ADAM

## Overview
ADAM now has comprehensive DBT (Data Build Tool) knowledge and can intelligently assist with creating production-ready DBT models following best practices.

## What We've Implemented

### 1. Knowledge Base Files
- **`DBT_KNOWLEDGE.md`** (12.8 KB) - Human-readable DBT best practices guide
- **`dbt_patterns.yaml`** (15.2 KB) - Structured patterns and templates for programmatic access

### 2. DBT Knowledge Service
- **Location**: `/src/adam_v2/services/dbt_knowledge_service.py`
- **Features**:
  - Detects DBT-related queries automatically
  - Provides relevant patterns and templates
  - Generates complete DBT models with proper configuration
  - Suggests improvements for existing models
  - Validates naming conventions
  - Includes Snowflake-specific optimizations

### 3. LLM Integration
- **Enhanced**: `/src/adam_v2/services/llm_service.py`
- Automatically detects DBT context in queries
- Enhances prompts with relevant DBT knowledge
- Works with both regular and streaming responses

### 4. Self-Learning Capability
- Stores successful DBT patterns in memory
- Learns from each model created
- Retrieves similar patterns for new requests
- Improves over time based on usage

## How to Use

### 1. Start the Backend
```bash
# Restart to load DBT knowledge
PYTHONPATH=/Users/vitoryago/ADAM/src python -m adam_v2.main
```

### 2. Example Queries for ADAM

#### Convert Looker PDT to DBT:
```
"Please convert this Looker PDT to a DBT model:
view: user_metrics {
  derived_table: {
    sql: SELECT user_id, COUNT(*) as events FROM events GROUP BY 1;;
  }
}"
```

#### Create Staging Model:
```
"Create a staging model for user data from the raw.users table in Snowflake"
```

#### Build Incremental Fact Table:
```
"Create an incremental fact table for user transactions with proper clustering"
```

#### Get Best Practices:
```
"What are DBT best practices for organizing models?"
```

#### Generate Macros:
```
"Write a DBT macro for safe division that prevents divide by zero errors"
```

## DBT Knowledge Coverage

### Layer Architecture
- **Staging**: Light transformations, one model per source
- **Intermediate**: Business logic, cross-source joins
- **Marts**: Business-ready facts and dimensions

### Naming Conventions
- `stg_<source>__<entity>` - Staging models
- `int_<entity>__<transformation>` - Intermediate models
- `fct_<entity>` - Fact tables
- `dim_<entity>` - Dimension tables

### Snowflake Optimizations
- Automatic clustering key recommendations
- Transient table settings for staging
- Time Travel configuration by layer
- Warehouse sizing for heavy transformations
- Result caching strategies

### Testing Patterns
- Schema tests (unique, not_null, relationships)
- Data quality tests (volume checks, freshness)
- Custom business logic validation

### Macro Library
- Safe division
- Surrogate key generation
- Incremental filters with lookback
- Date spine generation
- String cleaning utilities
- Dynamic pivoting

## Testing the Integration

Run the test script:
```bash
python test_dbt_integration.py
```

This will verify:
- DBT Knowledge Service functionality
- LLM integration
- Knowledge file availability
- Model generation capabilities

## Architecture

```
User Query → LLM Service → DBT Detection
                ↓
         DBT Knowledge Service
                ↓
    ┌──────────────────────────┐
    │  DBT_KNOWLEDGE.md         │
    │  dbt_patterns.yaml        │
    │  Memory (learned patterns)│
    └──────────────────────────┘
                ↓
         Enhanced Response
```

## Benefits

1. **Consistency**: All DBT models follow the same best practices
2. **Speed**: Instant generation of complex DBT patterns
3. **Learning**: ADAM improves based on successful patterns
4. **Documentation**: Automatic generation of model documentation
5. **Testing**: Built-in test recommendations
6. **Optimization**: Snowflake-specific performance enhancements

## Next Steps

1. **Use It**: Start asking ADAM to build DBT models
2. **Provide Feedback**: ADAM learns from successful patterns
3. **Extend Knowledge**: Add company-specific patterns to the knowledge base
4. **Monitor**: Check memory storage for learned patterns

## Example Output

When you ask ADAM to create a DBT model, it will generate:

```sql
{{ config(
    materialized='incremental',
    unique_key='transaction_id',
    cluster_by=['user_id', 'created_at'],
    on_schema_change='sync_all_columns'
) }}

WITH source_data AS (
    SELECT * FROM {{ ref('stg_source__transactions') }}
),

transformed AS (
    -- Business logic here
    SELECT
        transaction_id,
        user_id,
        amount,
        created_at
    FROM source_data
)

SELECT * FROM transformed
{% if is_incremental() %}
    WHERE created_at > (SELECT MAX(created_at) FROM {{ this }})
{% endif %}
```

## Troubleshooting

If DBT knowledge isn't working:
1. Check that knowledge files exist: `ls DBT_*.md dbt_patterns.yaml`
2. Restart the backend to reload knowledge
3. Check logs for "DBT Knowledge Service initialized"
4. Run the test script to verify integration

---

*DBT Knowledge Integration completed successfully! ADAM is now a DBT expert.* 🎉