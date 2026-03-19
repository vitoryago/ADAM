"""
DBT Documentation Generator
Automatically generates comprehensive documentation for DBT models
"""

import json
import yaml
from typing import Dict, List, Optional, Any
from datetime import datetime
from pathlib import Path
from collections import defaultdict
import re
import logging

from .parser import DBTModel, DBTManifestParser
from .lineage import DBTLineageTracker
from .optimizer import SQLOptimizer
from .yaml_updater import YAMLUpdater

logger = logging.getLogger(__name__)


class DBTDocGenerator:
    """
    Generates intelligent documentation for DBT projects
    """

    def __init__(self, parser: DBTManifestParser, lineage_tracker: DBTLineageTracker, optimizer: SQLOptimizer):
        self.parser = parser
        self.lineage_tracker = lineage_tracker
        self.optimizer = optimizer
        self.yaml_updater = YAMLUpdater()
        self.generated_docs = {}

    def generate_model_documentation(
        self,
        model_name: str,
        include_lineage: bool = True,
        include_tests: bool = True,
        include_optimizations: bool = True,
        platform: str = "redshift"
    ) -> str:
        """
        Generate comprehensive documentation for a single model
        """
        model = self._find_model_by_name(model_name)
        if not model:
            return f"Model {model_name} not found"

        doc = self._generate_model_header(model)
        doc += self._generate_model_description(model)

        if include_lineage:
            doc += self._generate_lineage_section(model)

        doc += self._generate_columns_section(model)

        if include_tests:
            doc += self._generate_tests_section(model)

        if include_optimizations:
            doc += self._generate_optimization_section(model, platform)

        doc += self._generate_usage_examples(model)
        doc += self._generate_maintenance_notes(model)

        self.generated_docs[model_name] = doc
        return doc

    def _find_model_by_name(self, name: str) -> Optional[DBTModel]:
        """Find a model by name"""
        for model in self.parser.models.values():
            if model.name == name:
                return model
        return None

    def _generate_model_header(self, model: DBTModel) -> str:
        """Generate the header section of documentation"""
        return f"""# {model.name}

## Overview
**Model ID**: `{model.unique_id}`
**Materialization**: `{model.materialization}`
**Database**: `{model.database or 'default'}`
**Schema**: `{model.schema or 'default'}`
**Full Path**: `{model.database or 'default'}.{model.schema or 'default'}.{model.name}`

**Tags**: {', '.join([f'`{tag}`' for tag in model.tags]) if model.tags else 'None'}
**Last Updated**: {datetime.now().strftime('%Y-%m-%d')}

---

"""

    def _generate_model_description(self, model: DBTModel) -> str:
        """Generate enhanced description with AI insights"""
        doc = "## Description\n\n"

        if model.description:
            doc += f"{model.description}\n\n"
        else:
            # Generate description from SQL analysis
            doc += self._analyze_sql_purpose(model.compiled_sql) + "\n\n"

        # Add business context
        doc += "### Business Context\n"
        doc += self._generate_business_context(model) + "\n\n"

        return doc

    def _analyze_sql_purpose(self, sql: str) -> str:
        """Analyze SQL to determine model purpose"""
        purpose = "This model "

        sql_upper = sql.upper()

        # Identify main operations
        if "UNION" in sql_upper:
            purpose += "combines data from multiple sources "
        elif "JOIN" in sql_upper:
            join_count = sql_upper.count("JOIN")
            purpose += f"integrates data from {join_count + 1} tables "
        elif "GROUP BY" in sql_upper:
            purpose += "aggregates data "
        else:
            purpose += "transforms data "

        # Identify data patterns
        if "DATE" in sql_upper or "TIMESTAMP" in sql_upper:
            purpose += "with time-series processing "
        if "SUM(" in sql_upper or "COUNT(" in sql_upper:
            purpose += "including metrics calculation "
        if "CASE WHEN" in sql_upper:
            purpose += "with conditional logic "

        return purpose.strip()

    def _generate_business_context(self, model: DBTModel) -> str:
        """Generate business context from model analysis"""
        context = []

        # Check model name patterns
        name_lower = model.name.lower()

        if "fact_" in name_lower or "fct_" in name_lower:
            context.append("This is a **fact table** containing transactional or event data")
        elif "dim_" in name_lower or "dimension_" in name_lower:
            context.append("This is a **dimension table** containing descriptive reference data")
        elif "staging_" in name_lower or "stg_" in name_lower:
            context.append("This is a **staging model** that prepares raw data for transformation")
        elif "int_" in name_lower or "intermediate_" in name_lower:
            context.append("This is an **intermediate model** used in multi-step transformations")
        elif "mart_" in name_lower or "reporting_" in name_lower:
            context.append("This is a **data mart** optimized for reporting and analytics")

        # Check materialization strategy
        if model.is_incremental:
            context.append("Uses **incremental loading** to efficiently process large datasets")
        elif model.materialization == "table":
            context.append("**Fully materialized** as a table for optimal query performance")
        elif model.materialization == "view":
            context.append("Implemented as a **view** for real-time data access")

        return "\n".join(f"- {c}" for c in context) if context else "- General purpose transformation model"

    def _generate_lineage_section(self, model: DBTModel) -> str:
        """Generate lineage documentation"""
        doc = "## Data Lineage\n\n"

        lineage = self.parser.get_model_lineage(model.name)

        # Upstream dependencies
        doc += "### Upstream Dependencies\n"
        if lineage["upstream"]:
            doc += "This model depends on:\n"
            for dep in lineage["upstream"]:
                dep_model = self.parser.models.get(dep)
                if dep_model:
                    doc += f"- **{dep_model.name}** (`{dep_model.materialization}`): {dep_model.description[:100] if dep_model.description else 'No description'}\n"
                else:
                    doc += f"- {dep}\n"
        else:
            doc += "*No upstream dependencies (source model)*\n"

        doc += "\n"

        # Downstream impact
        doc += "### Downstream Impact\n"
        if lineage["downstream"]:
            doc += "The following models depend on this model:\n"
            for dep in lineage["downstream"]:
                dep_model = self.parser.models.get(dep)
                if dep_model:
                    doc += f"- **{dep_model.name}**: {dep_model.description[:100] if dep_model.description else 'No description'}\n"
                else:
                    doc += f"- {dep}\n"
        else:
            doc += "*No downstream dependencies (terminal model)*\n"

        doc += "\n"
        return doc

    def _generate_columns_section(self, model: DBTModel) -> str:
        """Generate detailed column documentation"""
        doc = "## Columns\n\n"

        if not model.columns:
            doc += "*No column documentation available. Run `dbt docs generate` to populate column metadata.*\n\n"
            return doc

        doc += "| Column Name | Type | Description | Tests |\n"
        doc += "|-------------|------|-------------|-------|\n"

        for col_name, col_info in model.columns.items():
            col_type = col_info.get("data_type", "unknown")
            col_desc = col_info.get("description", "")
            col_tests = col_info.get("tests", [])

            test_badges = []
            for test in col_tests:
                if isinstance(test, dict):
                    test_name = list(test.keys())[0] if test else "test"
                else:
                    test_name = test
                test_badges.append(f"`{test_name}`")

            doc += f"| **{col_name}** | {col_type} | {col_desc} | {' '.join(test_badges) if test_badges else '-'} |\n"

        doc += "\n"

        # Add column statistics if available
        doc += "### Key Column Patterns\n"
        doc += self._analyze_column_patterns(model) + "\n"

        return doc

    def _analyze_column_patterns(self, model: DBTModel) -> str:
        """Analyze and document column patterns"""
        patterns = []

        for col_name, col_info in model.columns.items():
            col_lower = col_name.lower()

            # Identify common column types
            if col_lower.endswith("_id"):
                patterns.append(f"- `{col_name}`: Identifier/Foreign key")
            elif col_lower.endswith("_at"):
                patterns.append(f"- `{col_name}`: Timestamp column")
            elif col_lower.endswith("_date"):
                patterns.append(f"- `{col_name}`: Date column")
            elif col_lower.startswith("is_") or col_lower.startswith("has_"):
                patterns.append(f"- `{col_name}`: Boolean flag")
            elif col_lower.endswith("_amount") or col_lower.endswith("_price"):
                patterns.append(f"- `{col_name}`: Monetary value")
            elif col_lower.endswith("_count") or col_lower.endswith("_qty"):
                patterns.append(f"- `{col_name}`: Quantity/Count metric")

        return "\n".join(patterns) if patterns else "No specific patterns identified\n"

    def _generate_tests_section(self, model: DBTModel) -> str:
        """Generate test documentation"""
        doc = "## Data Quality Tests\n\n"

        model_tests = []
        for test_id, test in self.parser.tests.items():
            if model.unique_id in test.depends_on:
                model_tests.append(test)

        if not model_tests:
            doc += "*No tests configured for this model.*\n\n"
            doc += "### Recommended Tests\n"
            doc += self._suggest_tests(model) + "\n"
            return doc

        # Group tests by type
        generic_tests = [t for t in model_tests if t.test_type == "generic"]
        singular_tests = [t for t in model_tests if t.test_type == "singular"]

        if generic_tests:
            doc += "### Generic Tests\n"
            for test in generic_tests:
                doc += f"- **{test.name}**"
                if test.column_name:
                    doc += f" on `{test.column_name}`"
                doc += f" (Severity: {test.severity})\n"

        if singular_tests:
            doc += "\n### Custom Tests\n"
            for test in singular_tests:
                doc += f"- **{test.name}**: Custom validation logic\n"

        doc += "\n"
        return doc

    def _suggest_tests(self, model: DBTModel) -> str:
        """Suggest tests based on column patterns"""
        suggestions = []

        for col_name in model.columns.keys():
            col_lower = col_name.lower()

            if col_lower.endswith("_id") and col_lower != "id":
                suggestions.append(f"- Add `relationships` test for `{col_name}`")
            elif col_lower == "id" or col_lower.endswith("_key"):
                suggestions.append(f"- Add `unique` and `not_null` tests for `{col_name}`")
            elif col_lower.endswith("_at") or col_lower.endswith("_date"):
                suggestions.append(f"- Add `not_null` test for `{col_name}`")
            elif col_lower.startswith("is_") or col_lower.startswith("has_"):
                suggestions.append(f"- Add `accepted_values` test for `{col_name}` with values [true, false]")

        return "\n".join(suggestions[:5]) if suggestions else "- Add basic `not_null` tests for required fields\n"

    def _generate_optimization_section(self, model: DBTModel, platform: str) -> str:
        """Generate optimization recommendations"""
        doc = "## Performance Optimizations\n\n"

        # Get optimization suggestions
        suggestions = self.optimizer.analyze_sql(model.compiled_sql, platform)

        if not suggestions:
            doc += "*No optimization opportunities identified.*\n\n"
            return doc

        # Group by severity
        critical = [s for s in suggestions if s.severity == "CRITICAL"]
        high = [s for s in suggestions if s.severity == "HIGH"]
        medium = [s for s in suggestions if s.severity == "MEDIUM"]

        if critical:
            doc += "### 🔴 Critical Optimizations\n"
            for s in critical:
                doc += f"- **{s.title}**: {s.description}\n"
                doc += f"  ```sql\n  {s.suggested_pattern}\n  ```\n"

        if high:
            doc += "\n### 🟠 High Priority Optimizations\n"
            for s in high:
                doc += f"- **{s.title}**: {s.description}\n"

        if medium:
            doc += "\n### 🟡 Medium Priority Optimizations\n"
            for s in medium[:3]:  # Show only top 3 medium priority
                doc += f"- **{s.title}**: {s.description}\n"

        doc += "\n"
        return doc

    def _generate_usage_examples(self, model: DBTModel) -> str:
        """Generate usage examples for the model"""
        doc = "## Usage Examples\n\n"

        doc += "### Query Example\n"
        doc += f"""```sql
-- Basic query
SELECT *
FROM {model.database or 'database'}.{model.schema or 'schema'}.{model.name}
WHERE created_at >= CURRENT_DATE - 7
LIMIT 100;
```\n\n"""

        if model.is_incremental:
            doc += "### Incremental Run\n"
            doc += f"""```bash
# Run incremental update
dbt run --select {model.name}

# Full refresh if needed
dbt run --select {model.name} --full-refresh
```\n\n"""

        return doc

    def _generate_maintenance_notes(self, model: DBTModel) -> str:
        """Generate maintenance notes and recommendations"""
        doc = "## Maintenance Notes\n\n"

        notes = []

        # Check model characteristics
        if model.is_incremental:
            notes.append("- **Incremental Logic**: Review the incremental predicate regularly to ensure efficiency")
            notes.append("- **Full Refresh**: Schedule periodic full refreshes to maintain data integrity")

        if model.materialization == "table":
            notes.append("- **Table Maintenance**: Monitor table size and consider partitioning if growth is significant")
            notes.append("- **Statistics**: Run ANALYZE (Redshift) or automatic stats collection (Snowflake)")

        if len(model.depends_on) > 5:
            notes.append("- **Complex Dependencies**: Consider refactoring to reduce dependency complexity")

        if not model.description:
            notes.append("- **Documentation**: Add a description in the schema.yml file")

        if not notes:
            notes.append("- Model is well-maintained and documented")

        doc += "\n".join(notes) + "\n\n"

        doc += "### Monitoring Checklist\n"
        doc += "- [ ] Execution time trending within acceptable range\n"
        doc += "- [ ] Row count changes are expected\n"
        doc += "- [ ] No test failures in recent runs\n"
        doc += "- [ ] Dependencies are up to date\n\n"

        return doc

    def generate_project_overview(self) -> str:
        """Generate overview documentation for entire project"""
        doc = f"""# DBT Project Documentation

## Project Overview
**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Total Models**: {len(self.parser.models)}
**Total Sources**: {len(self.parser.sources)}
**Total Tests**: {len(self.parser.tests)}

## Model Statistics
"""

        # Materialization breakdown
        materializations = {}
        for model in self.parser.models.values():
            mat = model.materialization
            materializations[mat] = materializations.get(mat, 0) + 1

        doc += "### Materialization Types\n"
        for mat, count in materializations.items():
            doc += f"- **{mat}**: {count} models\n"

        # Test coverage
        coverage = self.parser.get_test_coverage()
        doc += f"\n### Test Coverage\n"
        doc += f"- **Models with tests**: {coverage['tested_models']}/{coverage['total_models']} ({coverage['coverage_percentage']:.1f}%)\n"
        doc += f"- **Total tests**: {coverage['total_tests']}\n"
        doc += f"- **Columns tested**: {coverage['columns_tested']}\n"

        return doc

    def export_to_yaml(self, model_name: str) -> str:
        """Export model documentation to dbt schema.yml format"""
        model = self._find_model_by_name(model_name)
        if not model:
            return ""

        # Build YAML structure
        yaml_doc = {
            "version": 2,
            "models": [
                {
                    "name": model.name,
                    "description": model.description or self._analyze_sql_purpose(model.compiled_sql),
                    "columns": []
                }
            ]
        }

        # Add columns
        for col_name, col_info in model.columns.items():
            column_doc = {
                "name": col_name,
                "description": col_info.get("description", "")
            }

            # Add suggested tests
            tests = []
            col_lower = col_name.lower()
            if col_lower.endswith("_id") and col_lower != "id":
                tests.append("not_null")
            elif col_lower == "id":
                tests.extend(["unique", "not_null"])

            if tests:
                column_doc["tests"] = tests

            yaml_doc["models"][0]["columns"].append(column_doc)

        return yaml.dump(yaml_doc, default_flow_style=False, sort_keys=False)

    def export_all_documentation(self, output_dir: str):
        """Export all generated documentation to files"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Generate docs for all models
        for model_id, model in self.parser.models.items():
            doc = self.generate_model_documentation(model.name)

            # Save as markdown
            doc_path = output_path / f"{model.name}.md"
            with open(doc_path, 'w') as f:
                f.write(doc)

        # Generate project overview
        overview = self.generate_project_overview()
        with open(output_path / "README.md", 'w') as f:
            f.write(overview)

        logger.info(f"Documentation exported to {output_dir}")

    def apply_suggestions_to_files(self, suggestions: Dict[str, Any], project_root: str) -> int:
        """
        Apply generated suggestions directly to schema.yml files
        
        Args:
            suggestions: Dictionary of suggestions (from export_suggestions_to_yaml)
            project_root: Path to project root
            
        Returns:
            Number of files updated
        """
        project_path = Path(project_root)
        updated_files = 0
        
        # Group suggestions by file path
        # Note: We need to find where each model is defined
        # This requires scanning the project structure
        
        # 1. Map models to their YAML files
        model_to_file = {}
        for yaml_file in project_path.rglob("*.yml"):
            if "dbt_project.yml" in yaml_file.name:
                continue
                
            try:
                with open(yaml_file) as f:
                    # We use simple load here just to map names, actual update uses YAMLUpdater
                    data = yaml.safe_load(f)
                    
                if data and "models" in data:
                    for model in data["models"]:
                        model_to_file[model["name"]] = yaml_file
            except Exception as e:
                logger.warning(f"Error reading {yaml_file}: {e}")
        
        # 2. Group updates by file
        file_updates = defaultdict(dict)
        
        for model_name, model_data in suggestions.items():
            if model_name in model_to_file:
                file_path = model_to_file[model_name]
                
                # Convert list of columns to dict for YAMLUpdater
                col_updates = {}
                for col in model_data.get("columns", []):
                    if "description" in col:
                        col_updates[col["name"]] = col["description"]
                
                if col_updates:
                    file_updates[file_path][model_name] = col_updates
        
        # 3. Apply updates
        for file_path, updates in file_updates.items():
            if self.yaml_updater.batch_update_descriptions(file_path, updates):
                updated_files += 1
                logger.info(f"Updated {file_path}")
                
        return updated_files