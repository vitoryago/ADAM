# DBT Knowledge Base for ADAM

> Production-Ready DBT Project Guide with Snowflake Optimizations
> Version: Latest DBT (1.7+) | Target: Snowflake

## Core Principles

1. **Clear Layer Separation**: Raw → Staging → Intermediate → Marts
2. **One Source, One Staging**: Each source table gets exactly one staging model
3. **Convention Over Configuration**: Folder structure determines materialization and schema
4. **Test Everything**: Data quality gates at every layer
5. **Document as You Build**: Documentation lives next to code

## Project Structure

```
dbt_analytics/
├── models/
│   ├── staging/          # Raw data entry point (one folder per source)
│   ├── intermediate/     # Business logic layer (organized by domain)
│   ├── marts/           # Business-ready datasets (organized by department)
│   └── utilities/       # Helper models and shared reference data
├── data/                # CSV seeds for static reference data
├── macros/              # Reusable SQL functions
├── snapshots/           # SCD Type 2 tracking
├── tests/               # Custom data tests
└── analyses/            # Ad-hoc queries and explorations
```

## Layer Architecture

### 1. Staging Layer
**Purpose**: Light transformation of raw data
**Naming**: `stg_<source>__<entity>.sql`
**Materialization**: `view` (default)
**Schema**: `staging`

#### What Belongs in Staging:
- Column renaming to standards
- Type casting (cents → dollars)
- JSON parsing/flattening
- Timezone standardization to UTC
- Test data filtering (`WHERE NOT test_mode`)
- Metadata addition (`_loaded_at`, `_batch_id`)

#### What Does NOT Belong:
- Multi-table joins
- Complex business logic
- Heavy aggregations
- Window functions

#### Snowflake Optimizations for Staging:
```sql
{{ config(
    materialized='view',
    schema='staging',
    transient=true,  -- Don't need Time Travel for staging
    secure=false     -- No need for secure views in staging
) }}
```

### 2. Intermediate Layer
**Purpose**: Business logic and cross-source joins
**Naming**: `int_<entity>__<transformation>.sql`
**Materialization**: `ephemeral` (default) or `table` if reused heavily
**Schema**: `intermediate`
**Organization**: By business domain (not source)

#### Examples:
- `int_users__with_acquisition_channel.sql`
- `int_campaigns__performance_metrics.sql`
- `int_attribution__last_touch.sql`

#### When to Materialize as Table:
- Used by 3+ downstream models
- Contains expensive aggregations
- DBT run time shows bottleneck

#### Snowflake Optimizations:
```sql
{{ config(
    materialized='table',
    cluster_by=['user_id', 'date_id'],  -- Cluster on common join/filter keys
    transient=false,  -- Keep for debugging
    post_hook="ALTER TABLE {{ this }} SET DATA_RETENTION_TIME_IN_DAYS = 7"
) }}
```

### 3. Marts Layer
**Purpose**: Analytics-ready, business-facing datasets
**Naming**: Facts: `fct_<entity>`, Dimensions: `dim_<entity>`
**Materialization**: `table` or `incremental`
**Schema**: Named by domain (`core`, `finance`, `marketing`)

#### Incremental Strategy for Facts:
```sql
{{ config(
    materialized='incremental',
    unique_key='advance_id',
    on_schema_change='sync_all_columns',
    cluster_by=['date_id', 'user_id'],
    incremental_strategy='merge',
    merge_update_columns=['status', 'updated_at']
) }}

SELECT * FROM source_data
{% if is_incremental() %}
    WHERE updated_at > (SELECT MAX(updated_at) FROM {{ this }})
      -- Add lookback for safety
      OR updated_at >= DATEADD('day', -3, CURRENT_DATE)
{% endif %}
```

## Naming Conventions

### Model Prefixes
- `stg_` : Staging models
- `int_` : Intermediate models
- `fct_` : Fact tables
- `dim_` : Dimension tables
- `rpt_` : Reporting/aggregate tables
- `snapshot_` : SCD Type 2 snapshots

### Double Underscore Convention
Use `__` to separate components:
- `stg_banking__payments` (source: banking, entity: payments)
- `int_users__with_lifetime_value` (entity: users, transformation: with_lifetime_value)

### File Organization
- `_<layer>__sources.yml` : Source definitions
- `_<layer>__models.yml` : Model configs and docs
- `_<layer>__docs.md` : Extended documentation

## Testing Strategy

### Schema Tests (in YAML)
```yaml
models:
  - name: fct_advances
    columns:
      - name: advance_id
        tests:
          - unique
          - not_null
      - name: user_id
        tests:
          - relationships:
              to: ref('dim_users')
              field: user_id
      - name: advance_amount
        tests:
          - dbt_expectations.expect_column_values_to_be_between:
              min_value: 50
              max_value: 500
              config:
                severity: warn
```

### Custom Data Tests
```sql
-- tests/assert_daily_volume_normal.sql
-- Returns rows only when anomaly detected
WITH daily_metrics AS (
    SELECT 
        date_id,
        COUNT(*) as row_count
    FROM {{ ref('fct_advances') }}
    WHERE date_id >= DATEADD('day', -30, CURRENT_DATE)
    GROUP BY date_id
),
stats AS (
    SELECT 
        AVG(row_count) as avg_count,
        STDDEV(row_count) as stddev_count
    FROM daily_metrics
)
SELECT * FROM daily_metrics
CROSS JOIN stats
WHERE date_id = CURRENT_DATE
  AND ABS(row_count - avg_count) > 3 * stddev_count
```

### Test Severity Levels
- `error`: Blocks deployment (primary keys, critical business rules)
- `warn`: Alerts but continues (statistical anomalies, soft limits)

## Macro Best Practices

### Only Create Macros for 3+ Uses
If used once: Keep as CTE in model
If used twice: Consider duplication
If used 3+ times: Create macro

### Essential Macro Library

#### Safe Division
```sql
{% macro safe_divide(numerator, denominator, precision=2) %}
    ROUND(
        CASE 
            WHEN {{ denominator }} = 0 OR {{ denominator }} IS NULL 
            THEN 0
            ELSE {{ numerator }}::FLOAT / {{ denominator }}
        END, 
        {{ precision }}
    )
{% endmacro %}
```

#### Generate Surrogate Key (Snowflake optimized)
```sql
{% macro generate_surrogate_key(fields) %}
    MD5_BINARY(
        CONCAT(
            {%- for field in fields %}
                COALESCE(CAST({{ field }} AS VARCHAR), 'null')
                {%- if not loop.last %}, '|', {% endif -%}
            {%- endfor %}
        )
    )
{% endmacro %}
```

#### Incremental Filter with Lookback
```sql
{% macro incremental_filter(timestamp_field, lookback_days=3) %}
    {% if is_incremental() %}
        WHERE {{ timestamp_field }} >= (
            SELECT DATEADD('day', -{{ lookback_days }}, 
                          MAX({{ timestamp_field }}))
            FROM {{ this }}
        )
    {% endif %}
{% endmacro %}
```

#### Date Spine Generator
```sql
{% macro generate_date_spine(start_date, end_date) %}
    WITH date_spine AS (
        SELECT 
            DATEADD('day', SEQ4(), '{{ start_date }}'::DATE) AS date_day
        FROM TABLE(GENERATOR(ROWCOUNT => 
            DATEDIFF('day', '{{ start_date }}'::DATE, '{{ end_date }}'::DATE) + 1
        ))
    )
    SELECT * FROM date_spine
{% endmacro %}
```

## Documentation Standards

### Model Documentation Template
```yaml
models:
  - name: model_name
    description: |
      Brief description of model purpose.
      **Grain**: One row per [entity] per [time period]
      **Source Systems**: [List sources]
      **Update Frequency**: [Daily/Hourly/Real-time]
      **Business Context**: [Why this exists]
      **Known Issues**: [Any gotchas]
    meta:
      owner: team_name
      sla: tier_1  # tier_1 (critical), tier_2 (important), tier_3 (nice-to-have)
      pii: true/false
    columns:
      - name: column_name
        description: Clear description
        meta:
          sensitivity: high/medium/low
```

### Column Description Standards
- For IDs: "Unique identifier for [entity]"
- For amounts: Include currency and whether tax-inclusive
- For dates: Specify timezone and whether it's event time or process time
- For enums: List all possible values with meanings

## Configuration Best Practices

### dbt_project.yml
```yaml
name: 'dbt_analytics'
version: '1.0.0'
config-version: 2

models:
  dbt_analytics:
    +on_schema_change: "sync_all_columns"
    +persist_docs:
      relation: true
      columns: true
    +transient: false  # Snowflake: Keep Time Travel for production
    
    staging:
      +materialized: view
      +schema: staging
      +tags: ['daily']
      
    intermediate:
      +materialized: ephemeral
      +schema: intermediate
      +tags: ['daily']
      
    marts:
      +materialized: table
      +tags: ['daily', 'critical']
      core:
        +schema: analytics_core
        +cluster_by: ['user_id']  # Default clustering
      finance:
        +schema: analytics_finance
        +cluster_by: ['date_id', 'account_id']
```

## Snowflake-Specific Optimizations

### 1. Clustering Keys
- Always cluster large fact tables (>1GB)
- Common patterns: `[date_column, high_cardinality_id]`
- Monitor with `SYSTEM$CLUSTERING_INFORMATION()`

### 2. Materialization Strategies
```sql
-- For large facts (>10M rows)
{{ config(
    materialized='incremental',
    unique_key='id',
    cluster_by=['date_id', 'user_id'],
    incremental_strategy='merge',
    transient=false,
    pre_hook="ALTER SESSION SET QUERY_TAG = 'dbt:{{ this }}'",
    post_hook=[
        "ALTER TABLE {{ this }} SET AUTO_CLUSTERING = TRUE",
        "GRANT SELECT ON {{ this }} TO ROLE ANALYTICS_READER"
    ]
) }}
```

### 3. Zero-Copy Cloning for Testing
```sql
{% macro clone_prod_for_testing() %}
    CREATE OR REPLACE TABLE {{ target.schema }}.test_{{ this.name }}
    CLONE {{ this }}
{% endmacro %}
```

### 4. Result Caching
```sql
-- Add to critical queries
ALTER SESSION SET USE_CACHED_RESULT = TRUE;
```

### 5. Warehouse Sizing
```yaml
# In profiles.yml
prod:
  outputs:
    default:
      warehouse: ANALYTICS_WH
      query_tag: 'dbt_prod'
      
# For heavy transformations
{{ config(
    pre_hook="ALTER SESSION SET WAREHOUSE = 'ANALYTICS_XL_WH'",
    post_hook="ALTER SESSION SET WAREHOUSE = 'ANALYTICS_WH'"
) }}
```

## Common Patterns

### SCD Type 2 Implementation
```sql
{% snapshot users_snapshot %}
{{ config(
    target_database='analytics',
    target_schema='snapshots',
    unique_key='user_id',
    strategy='timestamp',
    updated_at='updated_at',
    invalidate_hard_deletes=true
) }}
SELECT * FROM {{ ref('stg_app__users') }}
{% endsnapshot %}
```

### Union Multiple Sources
```sql
{% set sources = ['ios', 'android', 'web'] %}

WITH unioned AS (
    {% for source in sources %}
    SELECT 
        '{{ source }}' AS source_system,
        * 
    FROM {{ ref('stg_' ~ source ~ '__events') }}
    {% if not loop.last %}UNION ALL{% endif %}
    {% endfor %}
)
SELECT * FROM unioned
```

### Dynamic Pivot
```sql
{% macro pivot(column_name, values_list, agg_func='SUM') %}
    {% for value in values_list %}
    {{ agg_func }}(CASE WHEN {{ column_name }} = '{{ value }}' 
                   THEN amount ELSE 0 END) AS {{ value }}_amount
    {%- if not loop.last %},{% endif %}
    {% endfor %}
{% endmacro %}
```

## Orchestration Guidelines

### Development Workflow
1. Create feature branch
2. Develop models with `dbt run --select +my_model`
3. Test with `dbt test --select my_model`
4. Generate docs with `dbt docs generate`
5. Open PR with test results

### Production Jobs
```yaml
# Example Airflow DAG structure
1. Data Quality Checks (sources)
2. dbt run --select staging
3. dbt test --select staging
4. dbt run --select intermediate+
5. dbt run --select marts
6. dbt test --select marts
7. Grant permissions
8. Refresh BI tool extracts
```

### Monitoring & Alerts
- Set up alerts for test failures
- Monitor run times (>2hr indicates optimization needed)
- Track Snowflake credit usage per model
- Review query history for inefficient patterns

## Anti-Patterns to Avoid

1. **Direct source references in marts**: Always go through staging
2. **Business logic in staging**: Keep staging for technical transforms only
3. **Over-using ephemeral**: If referenced 3+ times, materialize it
4. **Missing incremental filters**: Always add lookback safety
5. **No documentation**: Undocumented models become technical debt
6. **Test-free models**: Every model should have at least basic tests
7. **Hardcoded values**: Use vars in dbt_project.yml
8. **Not using packages**: Check dbt-hub before writing custom macros

## Self-Learning Integration

When ADAM creates a new DBT model:
1. Store the pattern in memory with tags: `dbt_pattern`, source, complexity
2. Track success metrics: run time, test pass rate, user feedback
3. Surface similar patterns when creating new models
4. Build a library of company-specific patterns over time

---
*This knowledge base will be continuously updated as ADAM learns from successful DBT implementations*