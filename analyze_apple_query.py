#!/usr/bin/env python3
"""
Simple Apple Search Ads Query Builder
"""

# Your Apple Search Ads query for last 7 days best performing keywords
query = """
SELECT 
    date,
    campaign_name,
    keyword_name,
    keyword_cost,
    keyword_clicks,
    keyword_impressions,
    installs,
    su_lite,
    customer,
    -- Performance metrics
    ROUND(keyword_cost / NULLIF(keyword_clicks, 0), 2) AS cpc,
    ROUND(keyword_cost / NULLIF(installs, 0), 2) AS cpi,
    ROUND(100.0 * keyword_clicks / NULLIF(keyword_impressions, 0), 2) AS ctr_pct,
    ROUND(100.0 * installs / NULLIF(keyword_clicks, 0), 2) AS install_rate_pct,
    -- Value metrics
    ROUND(100.0 * customer / NULLIF(installs, 0), 2) AS customer_rate_pct,
    ROUND(keyword_cost / NULLIF(customer, 0), 2) AS cac
FROM 
    fct_daily_performance_marketing_keyword_level
WHERE 
    ad_platform = 'apple'
    AND date >= CURRENT_DATE - 7
    AND keyword_cost > 0
ORDER BY 
    -- Order by best performance (low CPI, high installs)
    CASE 
        WHEN installs > 0 THEN keyword_cost / installs 
        ELSE 999999 
    END ASC,
    installs DESC
LIMIT 100;
"""

print("🍎 Apple Search Ads - Last 7 Days Best Performing Keywords")
print("=" * 60)
print("\nQuery to run in your database:")
print(query)

print("\n📊 This query will show you:")
print("- Top 100 keywords by performance (lowest CPI with installs)")
print("- Key metrics: CPC, CPI, CTR%, Install Rate%")
print("- Customer conversion rate and CAC")
print("- Filtered to only keywords with spend in last 7 days")

print("\n💡 Additional queries you might want:")
print("\n1. Group by campaign performance:")
campaign_query = """
SELECT 
    campaign_name,
    SUM(keyword_cost) AS total_cost,
    SUM(installs) AS total_installs,
    SUM(customer) AS total_customers,
    ROUND(SUM(keyword_cost) / NULLIF(SUM(installs), 0), 2) AS campaign_cpi,
    ROUND(SUM(keyword_cost) / NULLIF(SUM(customer), 0), 2) AS campaign_cac
FROM 
    fct_daily_performance_marketing_keyword_level
WHERE 
    ad_platform = 'apple'
    AND date >= CURRENT_DATE - 7
    AND keyword_cost > 0
GROUP BY 
    campaign_name
ORDER BY 
    total_cost DESC;
"""
print(campaign_query)

print("\n2. Daily trend analysis:")
daily_trend_query = """
SELECT 
    date,
    SUM(keyword_cost) AS daily_cost,
    SUM(installs) AS daily_installs,
    ROUND(SUM(keyword_cost) / NULLIF(SUM(installs), 0), 2) AS daily_cpi
FROM 
    fct_daily_performance_marketing_keyword_level
WHERE 
    ad_platform = 'apple'
    AND date >= CURRENT_DATE - 7
GROUP BY 
    date
ORDER BY 
    date DESC;
"""
print(daily_trend_query)