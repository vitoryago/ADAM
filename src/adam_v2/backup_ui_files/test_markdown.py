#!/usr/bin/env python3
"""Test markdown rendering issue"""

from html import escape
import re

# Test the issue with backticks
test_content = """Here's the DBT model code:

```sql
-- This is SQL code
SELECT * FROM table
```

And some `inline code` too."""

print("Original content:")
print(test_content)
print("\n" + "="*50 + "\n")

# Escape HTML
escaped = escape(test_content)
print("After HTML escape:")
print(escaped)
print("\n" + "="*50 + "\n")

# Try the regex
pattern = r'```(\w*)\s*([\s\S]*?)```'
matches = re.findall(pattern, escaped)
print("Regex matches on escaped content:")
print(matches)

# Check if backticks are being escaped
print("\nBacktick character codes:")
print(f"Regular backtick: {ord('`')}")
if '`' in escaped:
    print("Backticks are NOT escaped")
else:
    print("Backticks ARE escaped")