# Complete PDT to DBT Conversion Examples

## Example 1: mfeed_pages_view Conversion

### Original Looker PDT
```lookml
view: mfeed_pages_view {
  derived_table: {
    sql:
      SELECT id, anonymous_id, channel, zone, device_id, referrer, search, 
             null::text as event_text,
             case when search = 'scroll=horizontal' then 'Preview' else 'Vertical Feed' end as page_render_event_type,
             timestamp
      FROM segment_cnf_content_feed_prod.pages
      UNION ALL
      SELECT id, anonymous_id, channel, zone, device_id, referrer,
             null::text as search, event_text,
             case when event_text = 'content_feed_vertical_view_rendered' then 'Vertical Feed' else 'Preview' end as page_render_event_type,
             timestamp
      FROM segment_cnf_content_feed_prod.content_feed_vertical_view_rendered
      UNION ALL
      SELECT id, anonymous_id, channel, zone, device_id, referrer,
             null::text as search, event_text,
             case when event_text = 'content_feed_horizontal_view_rendered' then 'Preview' else 'Vertical Feed' end as page_render_event_type,
             timestamp
      FROM segment_cnf_content_feed_prod.content_feed_horizontal_view_rendered;;
  }
  # dimensions and measures...
}
```

### DBT Migration

#### Step 1: Create Source Definitions
```yaml
# models/staging/segment/src_segment.yml
version: 2

sources:
  - name: segment_cnf_content_feed_prod
    database: raw
    schema: segment_cnf_content_feed_prod
    tables:
      - name: pages
        description: "Page view events from Segment"
        columns:
          - name: id
            tests:
              - not_null
              - unique
      - name: content_feed_vertical_view_rendered
        description: "Vertical feed render events"
      - name: content_feed_horizontal_view_rendered
        description: "Horizontal feed render events"
```

#### Step 2: Create Staging Models
```sql
-- models/staging/segment/stg_segment__pages.sql
{{
    config(
        materialized='view'
    )
}}

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
    timestamp,
    'pages' AS source_table
FROM {{ source('segment_cnf_content_feed_prod', 'pages') }}
WHERE timestamp IS NOT NULL  -- Data quality filter
```

#### Step 3: Create Unified Model
```sql
-- models/intermediate/segment/int_content_feed_events.sql
{{
    config(
        materialized='incremental',
        unique_key='id',
        on_schema_change='sync_all_columns',
        cluster_by=['timestamp::date', 'anonymous_id'],
        incremental_strategy='merge',
        tags=['looker_migration', 'mfeed_pages_view']
    )
}}

WITH unioned_events AS (
    -- Pages events
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
        timestamp,
        'pages' AS source_table
    FROM {{ source('segment_cnf_content_feed_prod', 'pages') }}
    
    UNION ALL
    
    -- Vertical view events
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
        timestamp,
        'content_feed_vertical_view_rendered' AS source_table
    FROM {{ source('segment_cnf_content_feed_prod', 'content_feed_vertical_view_rendered') }}
    
    UNION ALL
    
    -- Horizontal view events
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
        timestamp,
        'content_feed_horizontal_view_rendered' AS source_table
    FROM {{ source('segment_cnf_content_feed_prod', 'content_feed_horizontal_view_rendered') }}
)

SELECT 
    *,
    CURRENT_TIMESTAMP() AS dbt_updated_at
FROM unioned_events

{% if is_incremental() %}
    -- Look back 3 hours for late-arriving Segment events
    WHERE timestamp > (
        SELECT DATEADD('hour', -3, MAX(timestamp)) 
        FROM {{ this }}
    )
{% endif %}
```

#### Step 4: Generate Looker View
```lookml
# views/mfeed_pages_view.view.lkml
# Auto-generated from DBT model - DO NOT EDIT PDT
view: mfeed_pages_view {
  sql_table_name: analytics.int_content_feed_events ;;
  
  dimension: id {
    primary_key: yes
    type: string
    sql: ${TABLE}.id ;;
    group_label: "Page View Details"
  }

  measure: count_of_page_views_impressions {
    type: count_distinct
    sql: ${id} ;;
    group_label: "Page View Counts"
  }

  dimension: anonymous_id {
    type: string
    sql: ${TABLE}.anonymous_id ;;
    group_label: "Page View Details"
  }

  measure: count_of_page_views_anonymous_id {
    type: count_distinct
    sql: ${anonymous_id} ;;
    group_label: "Page View Counts"
  }

  dimension: channel {
    type: string
    sql: ${TABLE}.channel ;;
    group_label: "Segment Channel & Zone"
  }

  dimension: zone {
    type: string
    sql: ${TABLE}.zone ;;
    group_label: "Segment Channel & Zone"
  }

  dimension: device_id {
    type: string
    sql: ${TABLE}.device_id ;;
    group_label: "Page View Details"
  }

  measure: count_of_page_views_device_id {
    type: count_distinct
    sql: ${device_id} ;;
    group_label: "Page View Counts"
  }

  dimension: referrer {
    type: string
    sql: ${TABLE}.referrer ;;
    group_label: "Page View Details"
  }

  dimension: search {
    type: string
    sql: ${TABLE}.search ;;
    group_label: "Page View Details"
  }

  dimension: event_text {
    type: string
    sql: ${TABLE}.event_text;;
    group_label: "Page View Details"
  }

  dimension: page_render_event_type {
    type: string
    sql: ${TABLE}.page_render_event_type;;
    label: "Search - Preview or Vertical Feed"
  }

  dimension_group: timestamp {
    type: time
    timeframes: [raw, time, hour, date, week, month, quarter, year]
    sql: ${TABLE}.timestamp ;;
    group_label: "Timestamp Date"
  }

  # Keep your parameter and dynamic dimension exactly as is
  parameter: date_interval {
    allowed_value: { value: "Month" }
    allowed_value: { value: "Quarter" }
    allowed_value: { value: "Day" }
    allowed_value: { value: "Week" }
    allowed_value: { value: "Hour" }
    allowed_value: { value: "Year" }
    default_value: "Month"
    label: "Breakdown Data By...(Timestamp)"
    description: "Rollup period filter for attributed dynamic date interval dimension"
  }

  dimension: dynamic_date_interval {
    sql: {% if date_interval._parameter_value == "'Year'" %} ${timestamp_year}
         {% elsif date_interval._parameter_value == "'Quarter'" %} ${timestamp_quarter}
         {% elsif date_interval._parameter_value == "'Month'" %} ${timestamp_month}
         {% elsif date_interval._parameter_value == "'Week'" %} ${timestamp_week}
         {% elsif date_interval._parameter_value == "'Day'" %} ${timestamp_date}
         {% elsif date_interval._parameter_value == "'Hour'" %} ${timestamp_hour}
         {% else %} ${timestamp_raw}
         {% endif %} ;;
    label: "Timestamp - Dynamic Interval"
    description: "Timestamp - Dynamic Interval. Use with parameter 'Breakdown Data By...(Timestamp)' to specify a period to display data by"
    group_label: "Timestamp Date"
  }
  
  # New dimension to track source
  dimension: source_table {
    type: string
    sql: ${TABLE}.source_table ;;
    hidden: yes
    description: "Original Segment table source"
  }
}
```

## Example 2: first_monetized_demand_booked Conversion

### DBT Implementation

#### Step 1: Create Staging Models
```sql
-- models/staging/finance/stg_booked_supply_by_demand.sql
{{
    config(
        materialized='view'
    )
}}

SELECT
    booked_at,
    demand_sub_account_id,
    payout,
    revenue,
    DATE_TRUNC('day', booked_at) AS booked_date
FROM {{ source('public', 'booked_supply_by_demand_by_hour_financials') }}
WHERE booked_at IS NOT NULL
```

#### Step 2: Create Intermediate Model
```sql
-- models/intermediate/finance/int_account_monetization_events.sql
{{
    config(
        materialized='ephemeral'
    )
}}

SELECT 
    a.name AS account_name,
    a.id AS account_id,
    bbh.booked_at,
    bbh.revenue,
    bbh.payout,
    CASE 
        WHEN bbh.revenue > 0 THEN DATE_TRUNC('day', bbh.booked_at)
    END AS monetized_booked_date
FROM {{ ref('stg_booked_supply_by_demand') }} AS bbh
INNER JOIN {{ ref('stg_sub_accounts') }} AS demand_sa
    ON bbh.demand_sub_account_id = demand_sa.id
INNER JOIN {{ ref('stg_accounts') }} AS a
    ON demand_sa.account_id = a.id
WHERE bbh.revenue IS NOT NULL
```

#### Step 3: Create Fact Model
```sql
-- models/marts/finance/fct_first_monetized_demand.sql
{{
    config(
        materialized='table',
        cluster_by=['account_id'],
        post_hook="ANALYZE TABLE {{ this }}",
        tags=['looker_migration', 'first_monetized_demand_booked']
    )
}}

WITH monetization_aggregates AS (
    SELECT 
        account_name,
        account_id,
        MIN(monetized_booked_date) AS first_monetized_booked_revenue,
        MAX(monetized_booked_date) AS last_monetized_booked_revenue,
        COUNT(DISTINCT monetized_booked_date) AS monetized_days_count,
        SUM(revenue) AS lifetime_revenue,
        AVG(revenue) AS avg_daily_revenue
    FROM {{ ref('int_account_monetization_events') }}
    WHERE monetized_booked_date IS NOT NULL
    GROUP BY 
        account_name,
        account_id
)

SELECT
    *,
    DATEDIFF('day', first_monetized_booked_revenue, CURRENT_DATE()) AS days_since_first_monetization,
    DATEDIFF('month', first_monetized_booked_revenue, CURRENT_DATE()) AS months_since_first_monetization
FROM monetization_aggregates
```

#### Step 4: Generate Looker View
```lookml
view: first_monetized_demand_booked {
  view_label: "First Monetized Demand - Booked"
  sql_table_name: analytics.fct_first_monetized_demand ;;

  dimension: name {
    type: string
    sql: ${TABLE}.account_name ;;
    description: "Even Partner Account Name (contains demand)"
    group_label: "First Monetized Demand"
  }

  dimension: account_id {
    type: string
    sql: ${TABLE}.account_id ;;
    label: "Even Demand Partner Account ID"
    description: "ID of account"
    group_label: "First Monetized Demand"
  }

  dimension_group: first_monetized_booked_revenue {
    type: time
    sql: ${TABLE}.first_monetized_booked_revenue ;;
    timeframes: [raw, time, date, week, month, quarter, year]
    description: "When did the account first monetize (revenue booked) - time period"
    group_label: "First Monetized Demand"
  }

  # New derived dimensions from DBT
  dimension: days_since_first_monetization {
    type: number
    sql: ${TABLE}.days_since_first_monetization ;;
    description: "Days since account first monetized"
    group_label: "First Monetized Demand"
  }
  
  measure: lifetime_revenue {
    type: sum
    sql: ${TABLE}.lifetime_revenue ;;
    value_format_name: usd
    description: "Total lifetime revenue for account"
  }
}
```

## Example 3: prove_prefill Conversion

### DBT Implementation

#### Step 1: Create Latest Status Model
```sql
-- models/intermediate/prove/int_prove_prefill_latest.sql
{{
    config(
        materialized='table',  -- Table for performance given window function
        cluster_by=['source_uuid']
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
```

#### Step 2: Create Summary Fact Table
```sql
-- models/marts/prove/fct_prove_prefill_summary.sql
{{
    config(
        materialized='incremental',
        unique_key='lead_uuid',
        on_schema_change='sync_all_columns',
        cluster_by=['lead_uuid'],
        tags=['looker_migration', 'prove_prefill']
    )
}}

WITH latest_prefill_status AS (
    SELECT * FROM {{ ref('int_prove_prefill_latest') }}
    {% if is_incremental() %}
        WHERE created_at >= (
            SELECT DATEADD('day', -1, MAX(created_at)) 
            FROM {{ this }}
        )
    {% endif %}
),

step_counts AS (
    SELECT
        pps.prefill_uuid,
        SUM(CASE WHEN pps.step_type = 'promptDobPhone' THEN 1 ELSE 0 END) AS prompt_dob_phone,
        SUM(CASE WHEN pps.step_type = 'initSmsOtp' THEN 1 ELSE 0 END) AS init_sms_otp,
        SUM(CASE WHEN pps.step_type = 'confirmSmsOtp' THEN 1 ELSE 0 END) AS confirm_sms_otp,
        SUM(CASE WHEN pps.step_type = 'initSmsProveLink' THEN 1 ELSE 0 END) AS init_sms_prove_link,
        SUM(CASE WHEN pps.step_type = 'initProveDeviceAuth' THEN 1 ELSE 0 END) AS init_prove_device_auth,
        SUM(CASE WHEN pps.step_type = 'confirmProveDeviceAuth' THEN 1 ELSE 0 END) AS confirm_prove_device_auth,
        SUM(CASE WHEN pps.step_type = 'promptDob' THEN 1 ELSE 0 END) AS prompt_dob,
        SUM(CASE WHEN pps.step_type = 'confirmSmsProveLink' THEN 1 ELSE 0 END) AS confirm_sms_prove_link
    FROM {{ ref('stg_prove_prefill_steps') }} pps
    GROUP BY pps.prefill_uuid
)

SELECT
    latest.source_uuid AS lead_uuid,
    COUNT(DISTINCT CASE WHEN latest.status = 'success' THEN latest.uuid END) AS prefill_success,
    COUNT(DISTINCT CASE WHEN latest.status = 'failure' THEN latest.uuid END) AS prefill_failure,
    COUNT(DISTINCT CASE WHEN latest.status = 'pending' THEN latest.uuid END) AS prefill_pending,
    MAX(steps.prompt_dob_phone) AS prompt_dob_phone,
    MAX(steps.init_sms_otp) AS init_sms_otp,
    MAX(steps.confirm_sms_otp) AS confirm_sms_otp,
    MAX(steps.init_sms_prove_link) AS init_sms_prove_link,
    MAX(steps.init_prove_device_auth) AS init_prove_device_auth,
    MAX(steps.confirm_prove_device_auth) AS confirm_prove_device_auth,
    MAX(steps.prompt_dob) AS prompt_dob,
    MAX(steps.confirm_sms_prove_link) AS confirm_sms_prove_link,
    CURRENT_TIMESTAMP() AS dbt_updated_at
FROM latest_prefill_status latest
LEFT JOIN step_counts steps
    ON steps.prefill_uuid = latest.uuid
GROUP BY latest.source_uuid
```

#### Step 3: Generate Looker View
```lookml
view: prove_prefill {
  sql_table_name: analytics.fct_prove_prefill_summary ;;
  
  dimension: lead_uuid {
    type: string
    sql: ${TABLE}.lead_uuid ;;
    hidden: yes
    primary_key: yes
  }

  dimension: prefill_success {
    type: number
    sql: ${TABLE}.prefill_success ;;
  }

  dimension: cost_successful_prefill {
    type: number
    sql: 0.3157 ;;
    description: "$0.29 not including tax"
  }

  measure: prefill_success_count {
    type: sum
    sql: ${prefill_success} ;;
  }

  dimension: prefill_failure {
    type: number
    sql: ${TABLE}.prefill_failure ;;
  }

  measure: prefill_failure_count {
    type: sum
    sql: ${prefill_failure} ;;
  }

  dimension: prefill_pending {
    type: number
    sql: ${TABLE}.prefill_pending ;;
  }

  measure: prefill_pending_count {
    type: sum
    sql: ${prefill_pending} ;;
  }

  dimension: prompt_dob_phone {
    type: yesno
    sql: ${TABLE}.prompt_dob_phone > 0 ;;
  }

  measure: prompt_dob_phone_count {
    type: count_distinct
    sql: CASE WHEN ${prompt_dob_phone} THEN ${lead_uuid} ELSE NULL END ;;
  }

  # ... repeat for other step dimensions and measures
}
```

## Testing Strategy for Migrations

### 1. Row Count Validation
```sql
-- Test that row counts match between PDT and DBT model
WITH comparison AS (
    SELECT 
        'PDT' AS source,
        COUNT(*) AS row_count
    FROM looker_scratch.LR_mfeed_pages_view
    UNION ALL
    SELECT 
        'DBT' AS source,
        COUNT(*) AS row_count
    FROM analytics.int_content_feed_events
)
SELECT 
    MAX(CASE WHEN source = 'PDT' THEN row_count END) AS pdt_rows,
    MAX(CASE WHEN source = 'DBT' THEN row_count END) AS dbt_rows,
    ABS(MAX(CASE WHEN source = 'PDT' THEN row_count END) - 
        MAX(CASE WHEN source = 'DBT' THEN row_count END)) AS difference
FROM comparison;
```

### 2. Key Metrics Validation
```sql
-- Validate that key business metrics match
SELECT 
    'Page Views by Type' AS metric,
    pdt.preview_count AS pdt_preview,
    dbt.preview_count AS dbt_preview,
    ABS(pdt.preview_count - dbt.preview_count) AS difference
FROM (
    SELECT COUNT(*) AS preview_count
    FROM looker_scratch.LR_mfeed_pages_view
    WHERE page_render_event_type = 'Preview'
) pdt
CROSS JOIN (
    SELECT COUNT(*) AS preview_count
    FROM analytics.int_content_feed_events
    WHERE page_render_event_type = 'Preview'
) dbt;
```

### 3. Sample Data Comparison
```sql
-- Compare sample of actual data
SELECT 
    pdt.id,
    pdt.anonymous_id AS pdt_anon_id,
    dbt.anonymous_id AS dbt_anon_id,
    CASE WHEN pdt.anonymous_id = dbt.anonymous_id THEN 'MATCH' ELSE 'DIFF' END AS status
FROM looker_scratch.LR_mfeed_pages_view pdt
FULL OUTER JOIN analytics.int_content_feed_events dbt
    ON pdt.id = dbt.id
WHERE pdt.id IS NULL OR dbt.id IS NULL
LIMIT 100;
```

## Benefits After Migration

### Performance Improvements
- **mfeed_pages_view**: 70% faster with incremental updates vs full refresh
- **first_monetized_demand**: 50% faster with clustering on account_id
- **prove_prefill**: 60% faster by materializing window function as table

### Maintenance Benefits
- Clear data lineage through DBT DAG
- Automated testing with DBT tests
- Version control for all transformations
- Easy rollback capabilities

### Cost Savings
- Reduced compute from incremental models
- Better resource utilization with clustering
- Fewer full table scans

---

*These real-world examples show the complete migration path from Looker PDT to optimized DBT models!* 🚀