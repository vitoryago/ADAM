"""
DBT Column Documentation Service
Provides intelligent column documentation generation and management
"""

import logging
import json
from typing import Dict, List, Optional, Any
from pathlib import Path
import yaml

from ..dbt_analyzer.column_intelligence import ColumnIntelligenceEngine, ColumnInfo
from ..dbt_analyzer.parser import DBTManifestParser
from ..dbt_analyzer.direct_reader import DirectDBTReader
from ...adam.llm.client import UnifiedLLMClient

logger = logging.getLogger(__name__)

class DBTColumnDocumentationService:
    """
    Service for intelligent column documentation generation
    Integrates AI, pattern recognition, and cross-model analysis
    """

    def __init__(self):
        self.llm_client = UnifiedLLMClient()
        self.engines: Dict[str, ColumnIntelligenceEngine] = {}
        self.readers: Dict[str, DirectDBTReader] = {}
        self.parsers: Dict[str, DBTManifestParser] = {}

    def _get_or_create_engine(self, project_path: str) -> ColumnIntelligenceEngine:
        """Get or create a column intelligence engine for a project"""
        if project_path not in self.engines:
            # Try to use manifest first, fallback to direct reader
            parser = None
            reader = None

            manifest_path = Path(project_path) / "target" / "manifest.json"
            if manifest_path.exists():
                if project_path not in self.parsers:
                    self.parsers[project_path] = DBTManifestParser(str(manifest_path))
                    self.parsers[project_path].parse()
                parser = self.parsers[project_path]
            else:
                if project_path not in self.readers:
                    self.readers[project_path] = DirectDBTReader(project_path)
                    self.readers[project_path].read_project()
                reader = self.readers[project_path]

            self.engines[project_path] = ColumnIntelligenceEngine(parser=parser, reader=reader)

        return self.engines[project_path]

    async def analyze_columns(self, project_path: str, use_ai: bool = True) -> Dict[str, Any]:
        """
        Analyze all columns in a DBT project
        Returns comprehensive analysis with patterns, common columns, and suggestions
        """
        logger.info(f"Analyzing columns in project: {project_path}")

        engine = self._get_or_create_engine(project_path)
        column_catalog = engine.analyze_project_columns()

        # Generate report
        report = engine.generate_documentation_report()

        # If AI is enabled, enhance suggestions
        if use_ai:
            report = await self._enhance_with_ai(report, column_catalog, project_path)

        return report

    async def document_model_columns(self,
                                    project_path: str,
                                    model_name: str,
                                    use_ai: bool = True,
                                    preserve_existing: bool = True) -> Dict[str, Any]:
        """
        Generate column documentation for a specific model
        """
        logger.info(f"Generating column documentation for model: {model_name}")

        engine = self._get_or_create_engine(project_path)

        # Analyze if not already done
        if not engine.column_catalog:
            engine.analyze_project_columns()

        # Get columns for this model
        model_columns = {}
        for col_name, col_info in engine.column_catalog.items():
            if model_name in col_info.appears_in_models:
                description = None

                # Check for existing description
                if preserve_existing and model_name in col_info.existing_descriptions:
                    description = col_info.existing_descriptions[model_name]
                else:
                    # Generate description
                    if use_ai:
                        description = await self._generate_ai_description(
                            col_name, model_name, col_info, project_path
                        )
                    else:
                        description = engine.suggest_column_description(col_name, model_name)

                model_columns[col_name] = {
                    'name': col_name,
                    'description': description,
                    'pattern': col_info.pattern.pattern_type if col_info.pattern else None,
                    'is_foreign_key': col_info.is_foreign_key,
                    'references': f"{col_info.references_table}.{col_info.references_column}"
                                if col_info.references_table else None,
                    'appears_in_other_models': list(col_info.appears_in_models - {model_name})
                }

                # Add suggested tests
                tests = self._suggest_tests_for_column(col_info)
                if tests:
                    model_columns[col_name]['suggested_tests'] = tests

        return {
            'model': model_name,
            'columns': model_columns,
            'total_columns': len(model_columns),
            'documented_columns': sum(1 for c in model_columns.values() if c['description']),
            'pattern_columns': sum(1 for c in model_columns.values() if c['pattern']),
            'foreign_keys': sum(1 for c in model_columns.values() if c['is_foreign_key'])
        }

    async def find_common_columns(self,
                                 project_path: str,
                                 min_occurrences: int = 2) -> List[Dict[str, Any]]:
        """
        Find columns that appear in multiple models and need standardization
        """
        logger.info(f"Finding common columns in project: {project_path}")

        engine = self._get_or_create_engine(project_path)

        if not engine.column_catalog:
            engine.analyze_project_columns()

        common_columns = engine.find_common_columns(min_occurrences)

        results = []
        for col_name, col_info in common_columns:
            # Check for inconsistent descriptions
            unique_descriptions = set(col_info.existing_descriptions.values())
            needs_standardization = len(unique_descriptions) > 1

            result = {
                'column_name': col_name,
                'appears_in': list(col_info.appears_in_models),
                'occurrence_count': len(col_info.appears_in_models),
                'needs_standardization': needs_standardization,
                'existing_descriptions': col_info.existing_descriptions,
                'pattern_detected': col_info.pattern.pattern_type if col_info.pattern else None,
                'suggested_standard_description': await self._generate_ai_description(
                    col_name, None, col_info, project_path
                ) if needs_standardization else None
            }
            results.append(result)

        return results

    async def generate_schema_yml(self,
                                 project_path: str,
                                 model_path: Optional[str] = None,
                                 use_ai: bool = True,
                                 include_tests: bool = True) -> str:
        """
        Generate schema.yml content for models
        """
        logger.info(f"Generating schema.yml for project: {project_path}")

        engine = self._get_or_create_engine(project_path)

        if not engine.column_catalog:
            engine.analyze_project_columns()

        # Get suggestions
        suggestions = engine.export_suggestions_to_yaml()

        # Filter by model path if specified
        if model_path:
            model_name = Path(model_path).stem
            if model_name in suggestions:
                suggestions = {model_name: suggestions[model_name]}

        # Enhance with AI if enabled
        if use_ai:
            for model_name, model_def in suggestions.items():
                for column in model_def['columns']:
                    if not column.get('description'):
                        col_info = engine.column_catalog.get(column['name'])
                        if col_info:
                            column['description'] = await self._generate_ai_description(
                                column['name'], model_name, col_info, project_path
                            )

        # Convert to YAML
        schema = {
            'version': 2,
            'models': list(suggestions.values())
        }

        # Format as YAML
        yaml_content = yaml.dump(schema, default_flow_style=False, sort_keys=False)

        return yaml_content

    async def suggest_documentation_improvements(self,
                                                project_path: str,
                                                priority: str = 'high') -> List[Dict[str, Any]]:
        """
        Suggest documentation improvements prioritized by impact
        """
        logger.info(f"Suggesting documentation improvements for project: {project_path}")

        engine = self._get_or_create_engine(project_path)

        if not engine.column_catalog:
            engine.analyze_project_columns()

        improvements = []

        # Find undocumented common columns (high priority)
        common_columns = engine.find_common_columns(min_occurrences=3)
        for col_name, col_info in common_columns:
            if not col_info.existing_descriptions:
                improvements.append({
                    'type': 'undocumented_common_column',
                    'priority': 'high',
                    'column': col_name,
                    'appears_in': list(col_info.appears_in_models),
                    'impact': f"Used in {len(col_info.appears_in_models)} models",
                    'suggested_action': f"Add documentation for '{col_name}' across all models",
                    'suggested_description': engine.suggest_column_description(col_name)
                })

        # Find inconsistent descriptions (medium priority)
        for col_name, col_info in engine.column_catalog.items():
            unique_descriptions = set(col_info.existing_descriptions.values())
            if len(unique_descriptions) > 1:
                improvements.append({
                    'type': 'inconsistent_documentation',
                    'priority': 'medium',
                    'column': col_name,
                    'models_affected': list(col_info.appears_in_models),
                    'current_descriptions': col_info.existing_descriptions,
                    'suggested_action': f"Standardize documentation for '{col_name}'",
                    'suggested_description': engine.suggest_column_description(col_name)
                })

        # Find foreign keys without relationship tests (medium priority)
        for col_name, col_info in engine.column_catalog.items():
            if col_info.is_foreign_key and col_info.references_table:
                improvements.append({
                    'type': 'missing_relationship_test',
                    'priority': 'medium',
                    'column': col_name,
                    'appears_in': list(col_info.appears_in_models),
                    'references': f"{col_info.references_table}.{col_info.references_column}",
                    'suggested_action': f"Add relationship test for foreign key '{col_name}'"
                })

        # Filter by priority
        if priority != 'all':
            improvements = [i for i in improvements if i['priority'] == priority]

        # Sort by impact
        improvements.sort(key=lambda x: len(x.get('appears_in', [])), reverse=True)

        return improvements[:20]  # Return top 20 improvements

    async def _generate_ai_description(self,
                                      column_name: str,
                                      model_name: Optional[str],
                                      col_info: ColumnInfo,
                                      project_path: str) -> str:
        """
        Generate an AI-enhanced column description using LLM
        """
        # Build context for LLM
        context = f"Generate a concise, clear description for the database column '{column_name}'"

        if model_name:
            context += f" in the model '{model_name}'"

        if col_info.pattern:
            context += f". This appears to be a {col_info.pattern.pattern_type} column"

        if col_info.is_foreign_key and col_info.references_table:
            context += f". It's a foreign key referencing {col_info.references_table}.{col_info.references_column}"

        if len(col_info.appears_in_models) > 1:
            context += f". This column appears in {len(col_info.appears_in_models)} models"

        if col_info.existing_descriptions:
            existing = list(col_info.existing_descriptions.values())[0]
            context += f". Existing description: '{existing}'. Please improve or standardize it"

        context += ". Keep the description under 100 characters and make it business-friendly."

        try:
            # Use the LLM to generate description
            response = await self.llm_client.complete(
                prompt=context,
                model="claude-sonnet-4-5",  # Use Claude for documentation
                temperature=0.3,  # Lower temperature for consistency
                max_tokens=100
            )
            return response.content.strip()
        except Exception as e:
            logger.warning(f"AI description generation failed: {e}")
            # Fallback to pattern-based description
            engine = self._get_or_create_engine(project_path)
            return engine.suggest_column_description(column_name, model_name)

    async def _enhance_with_ai(self,
                              report: Dict,
                              column_catalog: Dict[str, ColumnInfo],
                              project_path: str) -> Dict:
        """
        Enhance the report with AI-generated insights
        """
        try:
            # Generate insights about the column patterns
            patterns_summary = json.dumps(report['patterns_detected'], indent=2)

            prompt = f"""Analyze these column patterns from a DBT project and provide insights:

{patterns_summary}

Provide 3-5 key insights about:
1. Data model maturity
2. Naming convention consistency
3. Missing documentation priorities
4. Potential data quality issues

Keep each insight concise (1-2 sentences)."""

            response = await self.llm_client.complete(
                prompt=prompt,
                model="claude-sonnet-4-5",
                temperature=0.5,
                max_tokens=300
            )

            insights = response.content.strip().split('\n')
            report['ai_insights'] = [i.strip() for i in insights if i.strip()]

        except Exception as e:
            logger.warning(f"AI enhancement failed: {e}")
            report['ai_insights'] = []

        return report

    def _suggest_tests_for_column(self, col_info: ColumnInfo) -> List:
        """Suggest DBT tests for a column based on its pattern"""
        tests = []

        if col_info.pattern:
            pattern_type = col_info.pattern.pattern_type

            if pattern_type == 'id':
                tests.extend(['unique', 'not_null'])
            elif pattern_type in ['timestamp', 'date']:
                tests.append('not_null')
            elif pattern_type == 'boolean':
                tests.extend(['not_null', {'accepted_values': {'values': [True, False]}}])
            elif pattern_type == 'status':
                tests.append('not_null')
            elif pattern_type == 'email':
                tests.append('not_null')
                # Could add custom email format test
            elif pattern_type in ['money', 'count']:
                tests.extend(['not_null', {'dbt_utils.expression_is_true': {
                    'expression': f'{col_info.name} >= 0'
                }}])

        if col_info.is_foreign_key and col_info.references_table:
            tests.append({
                'relationships': {
                    'to': f"ref('{col_info.references_table}')",
                    'field': col_info.references_column or 'id'
                }
            })

        return tests

    def clear_cache(self, project_path: Optional[str] = None):
        """Clear cached engines and readers"""
        if project_path:
            self.engines.pop(project_path, None)
            self.readers.pop(project_path, None)
            self.parsers.pop(project_path, None)
        else:
            self.engines.clear()
            self.readers.clear()
            self.parsers.clear()