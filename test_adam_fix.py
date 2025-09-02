def calculate_total(items):
    # Bug: TypeError will occur if items contains non-numeric values
    total = 0
    for item in items:
        total += item  # This will fail if item is not a number
    return total

# Test with mixed types - this will crash
result = calculate_total([1, 2, "three", 4])
print(f"Total: {result}")