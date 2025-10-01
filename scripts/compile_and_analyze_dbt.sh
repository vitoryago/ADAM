#!/bin/bash
# Helper script to compile and analyze DBT projects
# Usage: ./compile_and_analyze_dbt.sh /path/to/dbt/project [target]

set -e

PROJECT_PATH="${1:-.}"
TARGET="${2:-prod}"

echo "🔧 Compiling DBT project at: $PROJECT_PATH"
echo "   Using target: $TARGET"
echo ""

# Change to project directory
cd "$PROJECT_PATH"

# Check if dbt_project.yml exists
if [ ! -f "dbt_project.yml" ]; then
    echo "❌ Error: Not a DBT project (dbt_project.yml not found)"
    exit 1
fi

# Try to parse (doesn't need database connection)
echo "Attempting to parse project..."
if dbt parse --target "$TARGET" 2>&1 | tee /tmp/dbt_parse.log; then
    echo "✅ Parse successful!"
else
    echo "⚠️  Parse failed. Trying compile..."

    # Try compile if parse fails
    if dbt compile --target "$TARGET" 2>&1 | tee /tmp/dbt_compile.log; then
        echo "✅ Compile successful!"
    else
        echo "❌ Both parse and compile failed."
        echo ""
        echo "Common issues:"
        echo "  1. Missing credentials - Set required environment variables"
        echo "  2. Wrong target - Available targets in profiles.yml"
        echo "  3. Dependencies missing - Run 'dbt deps' first"
        echo ""
        echo "Last 10 lines of output:"
        tail -10 /tmp/dbt_compile.log
        exit 1
    fi
fi

# Check if manifest was created
if [ -f "target/manifest.json" ]; then
    echo ""
    echo "✅ manifest.json created successfully!"
    echo ""
    echo "Now analyzing with ADAM..."
    echo ""

    # Analyze with ADAM
    python -m adam_v2.dbt_analyzer.cli analyze "$PROJECT_PATH"
else
    echo "❌ manifest.json not found after compilation"
    exit 1
fi