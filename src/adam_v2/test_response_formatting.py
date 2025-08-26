#!/usr/bin/env python3
"""
Test script for response formatting improvements
"""

import asyncio
from services.response_formatter import ResponseFormatter

async def test_formatting():
    formatter = ResponseFormatter()
    
    # Test case 1: Badly formatted SQL from GPT-5
    bad_sql = """
Here's the dbt model:

```sql
{{ config
(
materialized
=
'table'
)
 }}
WITH
 latest_prefill_status 
AS
 
(

    
SELECT
 
*

    
FROM
 
(

        
SELECT

            pp
.
*
,

            ROW_NUMBER
(
)
 
OVER
 
(
PARTITION
 
BY
 pp
.
source_uuid 
ORDER
 
BY
 pp
.
created_at 
DESC
)
 
AS
 rn
        
FROM
 {{ source
(
'public'
,
 
'prove_prefill'
)
 }} pp
    
)
 sub
    
WHERE
 rn 
=
 
1

)

SELECT

    latest
.
source_uuid 
AS
 lead_uuid
,
    
COUNT
(
DISTINCT
 
CASE
 
WHEN
 latest
.
status
 
=
 
'success'
 
THEN
 latest
.
uuid 
END
)
 
AS
 prefill_success
FROM
 latest_prefill_status latest
GROUP
 
BY
 lead_uuid
```

This should fix the indentation issues.
"""

    # Test case 2: Python code with bad indentation
    bad_python = """
Here's the Python function:

```python
def
calculate_total
(
items
)
:
    
total
=
0
    
for
item
in
items
:
        
total
+=
item
.
price
    
return
total
```
"""

    # Test case 3: Incomplete response
    incomplete = """
Here's how to implement the feature:

1. First, create the database schema
2. Then implement the API endpoints
3. Finally, add the frontend compon...
"""

    print("Testing Response Formatter\n" + "="*50)
    
    # Test SQL formatting
    print("\n1. Testing SQL Formatting (GPT-5):")
    print("-" * 30)
    result = await formatter.format_response(bad_sql, "gpt-5")
    print("Was reformatted:", result.was_reformatted)
    print("Issues:", result.formatting_issues)
    print("\nFormatted output:")
    print(result.content[:500] + "..." if len(result.content) > 500 else result.content)
    
    # Test Python formatting
    print("\n2. Testing Python Formatting (GPT-5-mini):")
    print("-" * 30)
    result = await formatter.format_response(bad_python, "gpt-5-mini")
    print("Was reformatted:", result.was_reformatted)
    print("Issues:", result.formatting_issues)
    print("\nFormatted output:")
    print(result.content)
    
    # Test incomplete response detection
    print("\n3. Testing Incomplete Response Detection:")
    print("-" * 30)
    result = await formatter.format_response(incomplete, "gpt-5")
    print("Was truncated:", result.was_truncated)
    print("Issues:", result.formatting_issues)
    print("\nFormatted output:")
    print(result.content)
    
    # Test validation
    print("\n4. Testing SQL Validation:")
    print("-" * 30)
    validation = await formatter.validate_and_fix_response(bad_sql, "sql")
    print("Valid:", validation['valid'])
    print("Errors:", validation['errors'])

if __name__ == "__main__":
    asyncio.run(test_formatting())