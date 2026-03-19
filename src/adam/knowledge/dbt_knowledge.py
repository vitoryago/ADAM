"""
DBT Knowledge Service for ADAM
Provides intelligent DBT pattern retrieval, template generation,
project analysis, and conversational DBT assistance.

Consolidated from:
- dbt_knowledge_service.py (pattern retrieval, template generation)
- dbt_service.py (project ingestion, documentation, optimization via dbt_analyzer)
- dbt_assistant.py (natural language DBT queries, direct file reading)
- dbt_integration_service.py (chat integration, context formatting)
"""

import yaml
import os
import re
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums & Data Classes
# ---------------------------------------------------------------------------

class ModelLayer(Enum):
    STAGING = "staging"
    INTERMEDIATE = "intermediate"
    MART_FACT = "mart_fact"
    MART_DIMENSION = "mart_dimension"
    UTILITY = "utility"


class MaterializationStrategy(Enum):
    VIEW = "view"
    TABLE = "table"
    INCREMENTAL = "incremental"
    EPHEMERAL = "ephemeral"
    SNAPSHOT = "snapshot"


@dataclass
class DBTModelRequest:
    """Request for DBT model generation"""
    model_name: str
    layer: ModelLayer
    source_table: Optional[str] = None
    source_schema: Optional[str] = None
    business_logic: Optional[str] = None
    grain: Optional[str] = None
    unique_key: Optional[str] = None
    timestamp_column: Optional[str] = None
    cluster_keys: Optional[List[str]] = None
    domain: Optional[str] = None


@dataclass
class DBTPattern:
    """A DBT pattern or template"""
    name: str
    template: str
    config: Dict[str, Any]
    description: str
    example: Optional[str] = None


# ---------------------------------------------------------------------------
# Main Knowledge Service
# ---------------------------------------------------------------------------

class DBTKnowledgeService:
    """Service for retrieving and applying DBT best practices.

    Combines knowledge-base pattern retrieval, LLM-powered detection,
    project analysis (via dbt_analyzer), natural language query handling,
    and chat integration context formatting.
    """

    def __init__(self):
        """Initialize the DBT knowledge service"""
        self.knowledge_base_path = Path(__file__).parent.parent.parent.parent / "knowledge"
        self.patterns_file = self.knowledge_base_path / "dbt" / "dbt_patterns.yaml"
        self.knowledge_file = self.knowledge_base_path / "dbt" / "DBT_KNOWLEDGE.md"

        # Load patterns and knowledge
        self.patterns = self._load_patterns()
        self.knowledge = self._load_knowledge()

        # Pattern matchers for detecting DBT needs (more specific to avoid false positives)
        self.dbt_indicators = [
            'dbt', 'incremental model', 'jinja',
            'materialized=', 'ref(', 'source(', '{{ ', '{% ',
            'staging model', 'staging layer', 'intermediate model',
            'fact table', 'dimension table', 'data mart',
            'pdt', 'looker', 'derived_table', 'analytics engineering',
            'data transformation', 'data warehouse', 'snowflake model'
        ]

        # Integration keywords (from dbt_integration_service)
        self._dbt_keywords = [
            "dbt", "model", "models", "lineage", "upstream", "downstream",
            "materialization", "incremental", "table", "view",
            "ref(", "source(", "schema.yml", "dbt_project.yml",
            "compile", "test", "run", "snapshot"
        ]

        # ----- DBT Assistant state (from dbt_assistant.py) -----
        self._readers: Dict[str, Any] = {}
        self._current_project: Optional[str] = None

        # ----- DBT Service state (from dbt_service.py) -----
        self._parsers: Dict[str, Any] = {}
        self._memory_systems: Dict[str, Any] = {}
        self._lineage_trackers: Dict[str, Any] = {}
        self._optimizer = None  # Lazy-loaded

    # ------------------------------------------------------------------
    # Pattern / knowledge loading
    # ------------------------------------------------------------------

    def _load_patterns(self) -> Dict[str, Any]:
        """Load DBT patterns from YAML file"""
        if not self.patterns_file.exists():
            logger.warning(f"DBT patterns file not found at {self.patterns_file}")
            return {}

        try:
            with open(self.patterns_file, 'r') as f:
                patterns = yaml.safe_load(f)
                logger.info(f"Loaded DBT patterns from {self.patterns_file}")
                return patterns
        except Exception as e:
            logger.error(f"Error loading DBT patterns: {e}")
            return {}

    def _load_knowledge(self) -> str:
        """Load DBT knowledge from markdown file"""
        if not self.knowledge_file.exists():
            logger.warning(f"DBT knowledge file not found at {self.knowledge_file}")
            return ""

        try:
            with open(self.knowledge_file, 'r') as f:
                knowledge = f.read()
                logger.info(f"Loaded DBT knowledge from {self.knowledge_file}")
                return knowledge
        except Exception as e:
            logger.error(f"Error loading DBT knowledge: {e}")
            return ""

    # ------------------------------------------------------------------
    # Detection helpers
    # ------------------------------------------------------------------

    def detect_dbt_context(self, query: str) -> bool:
        """Detect if a query needs DBT knowledge using simple keyword check (fallback)"""
        query_lower = query.lower()
        return any(indicator in query_lower for indicator in self.dbt_indicators)

    def is_dbt_question(self, message: str, workspace_path: Optional[str] = None) -> bool:
        """Detect if the user is asking about DBT (integration-level check).

        Includes keyword match plus optional workspace model-name scanning.
        """
        message_lower = message.lower()

        # Check for explicit DBT keywords
        if any(keyword in message_lower for keyword in self._dbt_keywords):
            return True

        # Check if workspace has DBT projects with loaded model names
        if workspace_path:
            try:
                projects = self.detect_dbt_projects(workspace_path)
                if projects:
                    for project in projects[:2]:
                        project_path = project.get('path', '')
                        if project_path in self._readers:
                            reader = self._readers[project_path]
                            for model_name in list(reader.models.keys())[:100]:
                                if model_name.lower() in message_lower:
                                    return True
            except Exception as e:
                logger.warning(f"Error checking for DBT models: {e}")

        return False

    async def intelligent_dbt_detection(self, query: str) -> Tuple[bool, float]:
        """Use LLM to intelligently detect if query needs DBT knowledge"""
        try:
            from adam.llm.client import UnifiedLLMClient

            detection_prompt = """Analyze if this query requires DBT (Data Build Tool) knowledge or assistance.

Query: "{query}"

DBT-related queries include:
- Creating or converting data models (staging, intermediate, marts, facts, dimensions)
- Writing SQL transformations for analytics engineering
- Incremental materializations and strategies
- Jinja templating and macros
- Data testing and documentation
- Converting Looker PDTs or other BI tool models
- Data warehouse best practices for analytics
- Anything explicitly mentioning dbt, models, or data transformation pipelines

Non-DBT queries include:
- General programming questions
- Non-analytics SQL queries
- Machine learning or AI topics
- Web development
- System administration
- General conversation

Respond with ONLY a JSON object:
{{"needs_dbt": true/false, "confidence": 0.0-1.0, "reason": "brief explanation"}}
"""

            llm_client = UnifiedLLMClient()
            response = await llm_client.complete(
                prompt=detection_prompt.format(query=query),
                model="gpt-4o-mini",
                temperature=0.1,
                max_tokens=100
            )

            import json
            content = response.content.strip()

            # Clean JSON response
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()

            if not content.startswith('{'):
                json_start = content.find('{')
                if json_start != -1:
                    content = content[json_start:]

            if content.startswith('{'):
                brace_count = 0
                end_pos = 0
                for i, char in enumerate(content):
                    if char == '{':
                        brace_count += 1
                    elif char == '}':
                        brace_count -= 1
                        if brace_count == 0:
                            end_pos = i + 1
                            break
                if end_pos > 0:
                    content = content[:end_pos]

            result = json.loads(content)
            logger.info(
                f"DBT detection: {result['needs_dbt']} "
                f"(confidence: {result['confidence']:.2f}) - {result['reason']}"
            )
            return result['needs_dbt'], result['confidence']

        except Exception as e:
            logger.warning(f"Intelligent DBT detection failed, falling back to keywords: {e}")
            return self.detect_dbt_context(query), 0.5

    # ------------------------------------------------------------------
    # Layer / pattern helpers
    # ------------------------------------------------------------------

    def identify_model_layer(self, description: str) -> ModelLayer:
        """Identify which DBT layer a model belongs to"""
        description_lower = description.lower()

        if any(word in description_lower for word in ['staging', 'stg_', 'raw', 'source']):
            return ModelLayer.STAGING
        elif any(word in description_lower for word in ['intermediate', 'int_', 'transform']):
            return ModelLayer.INTERMEDIATE
        elif any(word in description_lower for word in ['fact', 'fct_', 'events', 'transactions']):
            return ModelLayer.MART_FACT
        elif any(word in description_lower for word in ['dimension', 'dim_', 'lookup', 'reference']):
            return ModelLayer.MART_DIMENSION
        else:
            return ModelLayer.UTILITY

    def get_layer_pattern(self, layer: ModelLayer) -> Dict[str, Any]:
        """Get pattern configuration for a specific layer"""
        layer_patterns = {
            ModelLayer.STAGING: self.patterns.get('layers', {}).get('staging', {}),
            ModelLayer.INTERMEDIATE: self.patterns.get('layers', {}).get('intermediate', {}),
            ModelLayer.MART_FACT: self.patterns.get('layers', {}).get('marts', {}).get('fact', {}),
            ModelLayer.MART_DIMENSION: self.patterns.get('layers', {}).get('marts', {}).get('dimension', {}),
        }
        return layer_patterns.get(layer, {})

    def get_macro(self, macro_name: str) -> Optional[str]:
        macros = self.patterns.get('macros', {})
        macro = macros.get(macro_name, {})
        return macro.get('template') if macro else None

    def get_test_pattern(self, test_type: str) -> Optional[str]:
        test_patterns = self.patterns.get('test_patterns', {})
        if test_type in test_patterns.get('basic_tests', []):
            return test_type
        quality_tests = test_patterns.get('data_quality_tests', {})
        if test_type in quality_tests:
            return quality_tests[test_type].get('template')
        return None

    def get_incremental_strategy(self, strategy_name: str) -> Dict[str, Any]:
        strategies = self.patterns.get('incremental_strategies', {})
        return strategies.get(strategy_name, strategies.get('merge_on_unique_key', {}))

    # ------------------------------------------------------------------
    # Model template generation
    # ------------------------------------------------------------------

    def generate_model_template(self, request: DBTModelRequest) -> str:
        """Generate a complete DBT model based on request"""
        layer_pattern = self.get_layer_pattern(request.layer)

        if not layer_pattern:
            return f"-- No pattern found for layer: {request.layer}"

        config = self._build_config(request, layer_pattern)
        sql = self._build_sql(request, layer_pattern.get('sql_template', ''))
        model = f"{config}\n\n{sql}"

        description = self._build_description(request, layer_pattern)
        model = f"-- {description}\n\n{model}"

        return model

    def _build_config(self, request: DBTModelRequest, pattern: Dict) -> str:
        config_template = pattern.get('config_template', '')

        if request.layer == ModelLayer.STAGING:
            materialization = 'view'
        elif request.layer == ModelLayer.INTERMEDIATE:
            materialization = 'ephemeral'
        elif request.layer == ModelLayer.MART_FACT and request.unique_key:
            materialization = 'incremental'
        else:
            materialization = 'table'

        cluster_keys = request.cluster_keys or ['id']
        cluster_keys_str = str(cluster_keys).replace("'", '"')
        merge_columns = "['updated_at', 'status']" if materialization == 'incremental' else "[]"

        config = config_template.format(
            materialization=materialization,
            unique_key=request.unique_key or 'id',
            cluster_keys=cluster_keys_str,
            domain=request.domain or 'core',
            source=request.source_schema or 'source',
            merge_columns=merge_columns
        )

        return config

    def _build_sql(self, request: DBTModelRequest, template: str) -> str:
        if request.layer == ModelLayer.STAGING:
            sql = f"""
WITH source AS (
    SELECT * FROM {{{{ source('{request.source_schema}', '{request.source_table}') }}}}
),

renamed AS (
    SELECT
        -- Rename and cast columns as needed
        id AS {request.model_name}_id,
        *,
        -- Add metadata
        CURRENT_TIMESTAMP() AS _loaded_at,
        '{{{{ invocation_id }}}}' AS _batch_id
    FROM source
    WHERE NOT test_mode  -- Filter test data
)

SELECT * FROM renamed"""

        elif request.layer == ModelLayer.INTERMEDIATE:
            sql = f"""
WITH base AS (
    SELECT * FROM {{{{ ref('{request.source_table or 'stg_model'}') }}}}
),

transformed AS (
    SELECT
        *,
        -- Apply business logic here
        {request.business_logic or '-- Business transformations'}
    FROM base
)

SELECT * FROM transformed"""

        elif request.layer == ModelLayer.MART_FACT:
            sql = f"""
WITH source_data AS (
    SELECT * FROM {{{{ ref('{request.source_table or 'int_model'}') }}}}
),

final AS (
    SELECT
        -- Primary Key
        {request.unique_key or 'id'} AS {request.unique_key or 'id'},

        -- Foreign Keys
        user_id,
        date_id,

        -- Measures
        amount,
        quantity,

        -- Metadata
        created_at,
        updated_at
    FROM source_data
)

SELECT * FROM final
{{% if is_incremental() %}}
    WHERE {request.timestamp_column or 'updated_at'} > (
        SELECT MAX({request.timestamp_column or 'updated_at'}) FROM {{{{ this }}}}
    )
{{% endif %}}"""

        else:
            sql = template or "-- Template SQL\nSELECT * FROM source"

        return sql

    def _build_description(self, request: DBTModelRequest, pattern: Dict) -> str:
        description_template = pattern.get('description_template', '')

        if not description_template:
            return f"Model: {request.model_name}"

        description = description_template.format(
            entity=request.model_name,
            source=request.source_schema or 'source',
            source_table=request.source_table or 'source_table',
            transformation=request.business_logic or 'transformation',
            grain=request.grain or 'one row per record',
            business_logic_description=request.business_logic or '',
            business_description=request.business_logic or 'Business transformation logic',
            dependencies=request.source_table or 'upstream models',
            frequency='daily',
            sla_tier='tier_2'
        )

        return description

    # ------------------------------------------------------------------
    # Knowledge retrieval
    # ------------------------------------------------------------------

    def get_relevant_knowledge(self, query: str) -> Dict[str, Any]:
        """Get all relevant DBT knowledge for a query"""
        query_lower = query.lower()

        knowledge_response: Dict[str, Any] = {
            'detected_need': [],
            'patterns': {},
            'macros': [],
            'best_practices': [],
            'examples': []
        }

        if 'incremental' in query_lower:
            knowledge_response['detected_need'].append('incremental')
            knowledge_response['patterns']['incremental'] = self.patterns.get('incremental_strategies', {})

        if any(word in query_lower for word in ['staging', 'stg']):
            knowledge_response['detected_need'].append('staging')
            knowledge_response['patterns']['staging'] = self.patterns.get('layers', {}).get('staging', {})

        if any(word in query_lower for word in ['macro', 'function']):
            knowledge_response['detected_need'].append('macros')
            knowledge_response['macros'] = list(self.patterns.get('macros', {}).keys())

        if 'test' in query_lower:
            knowledge_response['detected_need'].append('testing')
            knowledge_response['patterns']['testing'] = self.patterns.get('test_patterns', {})

        if any(word in query_lower for word in ['snowflake', 'optimization', 'cluster']):
            knowledge_response['detected_need'].append('snowflake')
            knowledge_response['patterns']['snowflake'] = self.patterns.get('snowflake_optimizations', {})

        if knowledge_response['detected_need']:
            sections = self._extract_relevant_sections(
                self.knowledge,
                knowledge_response['detected_need']
            )
            knowledge_response['best_practices'] = sections

        return knowledge_response

    def _extract_relevant_sections(self, knowledge: str, needs: List[str]) -> List[str]:
        sections: List[str] = []
        lines = knowledge.split('\n')
        current_section: List[str] = []
        capturing = False

        for line in lines:
            if line.startswith('#'):
                section_relevant = any(need in line.lower() for need in needs)
                if section_relevant:
                    capturing = True
                    if current_section:
                        sections.append('\n'.join(current_section))
                    current_section = [line]
                elif capturing and line.count('#') <= 2:
                    capturing = False
                    if current_section:
                        sections.append('\n'.join(current_section))
                    current_section = []
            elif capturing:
                current_section.append(line)

        if capturing and current_section:
            sections.append('\n'.join(current_section))

        return sections[:3]

    # ------------------------------------------------------------------
    # Suggestions & validation
    # ------------------------------------------------------------------

    def suggest_improvements(self, model_sql: str) -> List[str]:
        suggestions = []

        if 'select *' in model_sql.lower() and 'staging' not in model_sql.lower():
            suggestions.append("Avoid SELECT * in non-staging models. Explicitly list columns for clarity.")

        if '{{' not in model_sql:
            suggestions.append("Consider using DBT macros for reusable logic (ref(), source(), etc.)")

        if 'join' in model_sql.lower() and 'staging' in model_sql.lower():
            suggestions.append("Complex joins should be in intermediate layer, not staging")

        if 'incremental' in model_sql and 'unique_key' not in model_sql:
            suggestions.append("Incremental models should specify a unique_key in config")

        if len(model_sql.split('\n')) > 200:
            suggestions.append("Consider breaking this large model into smaller intermediate models")

        return suggestions

    def validate_model_name(self, name: str, layer: ModelLayer) -> bool:
        prefixes = {
            ModelLayer.STAGING: 'stg_',
            ModelLayer.INTERMEDIATE: 'int_',
            ModelLayer.MART_FACT: 'fct_',
            ModelLayer.MART_DIMENSION: 'dim_'
        }

        expected_prefix = prefixes.get(layer, '')
        return name.startswith(expected_prefix) if expected_prefix else True

    # ------------------------------------------------------------------
    # Context enhancement for LLM queries
    # ------------------------------------------------------------------

    def enhance_query_with_dbt_context(self, query: str) -> str:
        """Enhance a query with relevant DBT context"""
        if not self.detect_dbt_context(query):
            return query

        knowledge = self.get_relevant_knowledge(query)
        context_parts: List[str] = []

        context_parts.append("[DBT Knowledge Base - Snowflake Optimized]")
        context_parts.append("=" * 50)

        context_parts.append("\nDBT Best Practices (from your knowledge base):")
        context_parts.append("- Layer Architecture: Staging -> Intermediate -> Marts")
        context_parts.append("- Naming: stg_<source>__<entity>, int_<entity>__<transform>, fct_/dim_")
        context_parts.append("- Staging: Views, light transforms only (rename, cast, filter)")
        context_parts.append("- Intermediate: Ephemeral, business logic, cross-source joins")
        context_parts.append("- Marts: Tables/Incremental, business-ready, clustered")

        context_parts.append("\nSnowflake Optimizations:")
        context_parts.append("- Cluster large tables (>1GB) by [date_column, high_cardinality_id]")
        context_parts.append("- Use transient=true for staging (no Time Travel needed)")
        context_parts.append("- Set AUTO_CLUSTERING = TRUE for fact tables")
        context_parts.append("- Warehouse sizing: Use XL for heavy transforms")

        if 'incremental' in knowledge['detected_need']:
            context_parts.append("\nIncremental Patterns:")
            if 'incremental' in knowledge['patterns']:
                strategies = knowledge['patterns']['incremental']
                for strategy_name, strategy_data in list(strategies.items())[:2]:
                    context_parts.append(f"- {strategy_name}: {strategy_data.get('description', '')}")
            context_parts.append("- Always add lookback safety (3 days)")
            context_parts.append("- Use merge strategy for updates, append for immutable events")

        if 'test' in query.lower():
            context_parts.append("\nTesting Strategy:")
            context_parts.append("- Schema tests: unique, not_null, relationships")
            context_parts.append("- Custom tests: volume anomalies, freshness checks")
            context_parts.append("- Severity: error (blocks deploy), warn (alerts)")

        if 'macro' in query.lower() and knowledge['macros']:
            context_parts.append(f"\nAvailable Macros: {', '.join(knowledge['macros'][:5])}")

        if knowledge['best_practices']:
            context_parts.append("\nDetailed Best Practices:")
            for section in knowledge['best_practices'][:2]:
                section_lines = section.split('\n')[:20]
                context_parts.append('\n'.join(section_lines))

        if knowledge['patterns']:
            context_parts.append("\nRemember:")
            context_parts.append("- One source = one staging model")
            context_parts.append("- Test everything (100% coverage on keys)")
            context_parts.append("- Document as you build")
            context_parts.append("- Use ref() and source() always")

        if context_parts:
            context = "\n".join(context_parts)
            enhanced = f"{query}\n\n{context}"
            logger.info(f"Enhanced query from {len(query)} to {len(enhanced)} chars")
            return enhanced

        return query

    # ------------------------------------------------------------------
    # Project discovery & loading (from dbt_assistant / dbt_service)
    # ------------------------------------------------------------------

    def detect_dbt_projects(self, workspace_path: str) -> List[Dict[str, Any]]:
        """Find all DBT projects in workspace"""
        try:
            projects: List[Dict[str, Any]] = []
            workspace = Path(workspace_path)

            for dbt_project_file in workspace.rglob("dbt_project.yml"):
                project_dir = dbt_project_file.parent
                models_dir = project_dir / "models"
                has_models = models_dir.exists()
                model_count = len(list(models_dir.rglob("*.sql"))) if has_models else 0
                manifest_path = project_dir / "target" / "manifest.json"

                projects.append({
                    "path": str(project_dir),
                    "name": project_dir.name,
                    "has_models": has_models,
                    "model_count": model_count,
                    "has_manifest": manifest_path.exists(),
                })

            return projects

        except Exception as e:
            logger.error(f"Error detecting DBT projects: {e}")
            return []

    def load_project(self, project_path: str, force_reload: bool = False) -> Dict[str, Any]:
        """Load a DBT project via DirectDBTReader (no manifest needed)"""
        try:
            if project_path in self._readers and not force_reload:
                return {"success": True, "message": "Project already loaded", "cached": True}

            from adam.knowledge.dbt_analyzer.direct_reader import DirectDBTReader

            reader = DirectDBTReader(project_path)
            if not reader.read_project():
                return {"success": False, "error": "Failed to read DBT project. Check if dbt_project.yml exists."}

            self._readers[project_path] = reader
            self._current_project = project_path

            stats = reader.get_project_stats()
            return {
                "success": True,
                "project_name": stats["project_name"],
                "project_path": project_path,
                "statistics": stats,
                "cached": False,
            }

        except Exception as e:
            logger.error(f"Error loading project: {e}")
            return {"success": False, "error": str(e)}

    # ------------------------------------------------------------------
    # Natural-language query handling (from dbt_assistant)
    # ------------------------------------------------------------------

    def handle_query(self, query: str, project_path: Optional[str] = None) -> Dict[str, Any]:
        """Handle natural language queries about DBT models"""
        if project_path:
            if project_path not in self._readers:
                load_result = self.load_project(project_path)
                if not load_result["success"]:
                    return load_result
            reader = self._readers[project_path]
        else:
            reader = self._readers.get(self._current_project) if self._current_project else None
            if not reader:
                return {"success": False, "error": "No DBT project loaded. Load a project first."}

        query_lower = query.lower()

        if "tell me about" in query_lower or "what is" in query_lower or "explain" in query_lower:
            model_name = self._extract_model_name(query, reader)
            if model_name:
                return self._describe_model(reader, model_name)

        elif "depends on" in query_lower or "upstream" in query_lower or "uses" in query_lower:
            model_name = self._extract_model_name(query, reader)
            if model_name:
                return self._get_dependencies(reader, model_name)

        elif "used by" in query_lower or "downstream" in query_lower or "impacts" in query_lower:
            model_name = self._extract_model_name(query, reader)
            if model_name:
                return self._get_downstream(reader, model_name)

        elif "complex" in query_lower:
            return self._get_complex_models(reader)

        elif "optimize" in query_lower or "improve" in query_lower:
            model_name = self._extract_model_name(query, reader)
            if model_name:
                platform = "snowflake" if "snowflake" in query_lower else "redshift"
                return self._get_optimizations(reader, model_name, platform)

        elif "search" in query_lower or "find" in query_lower:
            search_term = self._extract_search_term(query)
            if search_term:
                return self._search_models(reader, search_term)

        elif "how many" in query_lower or "count" in query_lower or "stats" in query_lower:
            return self._get_stats(reader)

        elif "list" in query_lower or "show all" in query_lower:
            if "test" in query_lower:
                return self._list_models_with_tests(reader)
            else:
                return self._list_models(reader)

        # Default: try to extract model name and describe
        model_name = self._extract_model_name(query, reader)
        if model_name:
            return self._describe_model(reader, model_name)

        return {
            "success": False,
            "error": (
                "I didn't understand your query. Try asking about a specific model, "
                "or use commands like 'tell me about', 'depends on', 'used by', 'optimize', etc."
            )
        }

    # ----- Private helpers for handle_query -----

    def _extract_model_name(self, query: str, reader: Any) -> Optional[str]:
        query_lower = query.lower()
        for model_name in reader.models.keys():
            if model_name.lower() in query_lower:
                return model_name
        matches = re.findall(r'["\']([^"\']+)["\']', query)
        if matches:
            potential_name = matches[0]
            if potential_name in reader.models:
                return potential_name
        return None

    @staticmethod
    def _extract_search_term(query: str) -> Optional[str]:
        patterns = [
            r'search\s+for\s+["\']?([^"\']+)["\']?',
            r'find\s+["\']?([^"\']+)["\']?',
            r'models\s+with\s+["\']?([^"\']+)["\']?',
        ]
        for pattern in patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        return None

    @staticmethod
    def _describe_model(reader: Any, model_name: str) -> Dict[str, Any]:
        model = reader.get_model(model_name)
        if not model:
            return {"success": False, "error": f"Model '{model_name}' not found"}
        upstream = reader.get_upstream_models(model_name)
        downstream = reader.get_downstream_models(model_name)
        return {
            "success": True,
            "type": "model_description",
            "model": {
                "name": model.name,
                "path": model.path,
                "description": model.description or "No description available",
                "complexity_score": model.complexity_score,
                "materialization": model.materialization or "view",
                "upstream_models": upstream,
                "downstream_models": downstream,
                "num_ctes": len(model.ctes),
                "uses_window_functions": model.uses_window_functions,
                "uses_joins": model.uses_joins,
                "uses_aggregations": model.uses_aggregations,
                "tests": model.tests,
                "columns": model.columns,
                "sources": [f"{s[0]}.{s[1]}" for s in model.sources]
            }
        }

    @staticmethod
    def _get_dependencies(reader: Any, model_name: str) -> Dict[str, Any]:
        model = reader.get_model(model_name)
        if not model:
            return {"success": False, "error": f"Model '{model_name}' not found"}
        upstream = reader.get_upstream_models(model_name)
        dependencies = []
        for dep_name in upstream:
            dep_model = reader.get_model(dep_name)
            if dep_model:
                dependencies.append({
                    "name": dep_name,
                    "complexity": dep_model.complexity_score,
                    "materialization": dep_model.materialization or "view"
                })
        return {
            "success": True,
            "type": "dependencies",
            "model": model_name,
            "upstream_models": dependencies,
            "sources": [f"{s[0]}.{s[1]}" for s in model.sources]
        }

    @staticmethod
    def _get_downstream(reader: Any, model_name: str) -> Dict[str, Any]:
        downstream = reader.get_downstream_models(model_name)
        downstream_models = []
        for dep_name in downstream:
            dep_model = reader.get_model(dep_name)
            if dep_model:
                downstream_models.append({
                    "name": dep_name,
                    "complexity": dep_model.complexity_score,
                    "materialization": dep_model.materialization or "view"
                })
        return {
            "success": True,
            "type": "downstream",
            "model": model_name,
            "downstream_models": downstream_models,
            "count": len(downstream_models)
        }

    @staticmethod
    def _get_complex_models(reader: Any, threshold: int = 15) -> Dict[str, Any]:
        complex_models = [
            {
                "name": model.name,
                "complexity": model.complexity_score,
                "num_ctes": len(model.ctes),
                "uses_window_functions": model.uses_window_functions,
                "materialization": model.materialization or "view"
            }
            for model in reader.models.values()
            if model.complexity_score >= threshold
        ]
        complex_models.sort(key=lambda x: x["complexity"], reverse=True)
        return {
            "success": True,
            "type": "complex_models",
            "threshold": threshold,
            "models": complex_models[:20],
            "total_count": len(complex_models)
        }

    def _get_optimizations(self, reader: Any, model_name: str, platform: str) -> Dict[str, Any]:
        model = reader.get_model(model_name)
        if not model:
            return {"success": False, "error": f"Model '{model_name}' not found"}

        from adam.knowledge.dbt_analyzer.optimizer import SQLOptimizer
        if self._optimizer is None:
            self._optimizer = SQLOptimizer()

        suggestions = self._optimizer.analyze_sql(model.sql, platform)
        return {
            "success": True,
            "type": "optimizations",
            "model": model_name,
            "platform": platform,
            "suggestions": [
                {
                    "severity": s.severity,
                    "category": s.category,
                    "title": s.title,
                    "description": s.description,
                    "estimated_impact": s.estimated_impact
                }
                for s in suggestions
            ]
        }

    @staticmethod
    def _search_models(reader: Any, search_term: str) -> Dict[str, Any]:
        results = []
        search_lower = search_term.lower()
        for name, model in reader.models.items():
            if search_lower in name.lower():
                results.append({"name": name, "match_type": "name", "description": model.description or "No description"})
                continue
            if model.description and search_lower in model.description.lower():
                results.append({"name": name, "match_type": "description", "description": model.description[:100]})
                continue
            if search_lower in model.sql.lower():
                results.append({"name": name, "match_type": "sql", "description": model.description or "Found in SQL"})
        return {
            "success": True,
            "type": "search_results",
            "query": search_term,
            "results": results[:20],
            "total_count": len(results)
        }

    @staticmethod
    def _get_stats(reader: Any) -> Dict[str, Any]:
        stats = reader.get_project_stats()
        return {"success": True, "type": "statistics", "stats": stats}

    @staticmethod
    def _list_models(reader: Any, limit: int = 50) -> Dict[str, Any]:
        models = [
            {
                "name": model.name,
                "complexity": model.complexity_score,
                "materialization": model.materialization or "view",
                "description": model.description or "No description"
            }
            for model in sorted(reader.models.values(), key=lambda m: m.complexity_score, reverse=True)
        ]
        return {"success": True, "type": "model_list", "models": models[:limit], "total_count": len(models)}

    @staticmethod
    def _list_models_with_tests(reader: Any) -> Dict[str, Any]:
        models = [
            {"name": model.name, "test_count": len(model.tests), "tests": model.tests}
            for model in reader.models.values()
            if model.tests
        ]
        return {"success": True, "type": "models_with_tests", "models": models, "total_count": len(models)}

    # ------------------------------------------------------------------
    # DBT context for chat integration (from dbt_integration_service)
    # ------------------------------------------------------------------

    def get_dbt_context(self, message: str, workspace_path: Optional[str] = None) -> str:
        """Get DBT context to add to the conversation"""
        if not workspace_path:
            return ""
        try:
            project_path = self._ensure_project_loaded(workspace_path)
            if not project_path:
                return ""
            result = self.handle_query(message, project_path)
            if not result.get('success'):
                return ""
            return self._format_dbt_result(result)
        except Exception as e:
            logger.error(f"Error getting DBT context: {e}")
            return ""

    def _ensure_project_loaded(self, workspace_path: str) -> Optional[str]:
        if self._current_project:
            return self._current_project
        projects = self.detect_dbt_projects(workspace_path)
        if not projects:
            return None
        for project in projects:
            if project.get('model_count', 0) > 0:
                result = self.load_project(project['path'])
                if result['success']:
                    logger.info(f"Auto-loaded DBT project: {project['name']}")
                    return project['path']
        return None

    @staticmethod
    def _format_dbt_result(result: Dict[str, Any]) -> str:
        query_type = result.get('type', 'unknown')

        if query_type == 'model_description':
            model = result['model']
            context = f"\n\n=== DBT Model: {model['name']} ===\n"
            context += f"Description: {model['description']}\n"
            context += f"Complexity: {model['complexity_score']}\n"
            context += f"Materialization: {model['materialization']}\n"
            if model['upstream_models']:
                context += f"Depends on: {', '.join(model['upstream_models'][:5])}\n"
            if model['downstream_models']:
                context += f"Used by: {', '.join(model['downstream_models'][:5])}\n"
            if model['tests']:
                context += f"Tests: {len(model['tests'])} tests defined\n"
            if model['uses_window_functions']:
                context += "Uses window functions\n"
            return context

        elif query_type == 'dependencies':
            model_name = result['model']
            upstream = result['upstream_models']
            context = f"\n\n=== Dependencies for {model_name} ===\n"
            if upstream:
                context += f"Upstream models ({len(upstream)}):\n"
                for dep in upstream[:10]:
                    context += f"  - {dep['name']} (complexity: {dep['complexity']})\n"
            if result['sources']:
                context += f"Sources: {', '.join(result['sources'])}\n"
            return context

        elif query_type == 'statistics':
            stats = result['stats']
            context = f"\n\n=== DBT Project Statistics ===\n"
            context += f"Project: {stats['project_name']}\n"
            context += f"Total models: {stats['total_models']}\n"
            context += f"Total sources: {stats['total_sources']}\n"
            return context

        return ""


# ---------------------------------------------------------------------------
# Memory Integration for self-learning
# ---------------------------------------------------------------------------

class DBTMemoryIntegration:
    """Store and learn from successful DBT patterns"""

    def __init__(self, project_id: str):
        self.project_id = project_id

    async def store_successful_pattern(
        self,
        model_name: str,
        model_sql: str,
        execution_time: float,
        test_results: Dict[str, bool]
    ):
        """Store a successful DBT pattern in memory"""
        from adam.memory.project import ProjectAwareMemory

        memory_service = ProjectAwareMemory(self.project_id, "dbt_patterns")

        test_pass_rate = sum(test_results.values()) / len(test_results) if test_results else 0
        quality_score = test_pass_rate * (1 / max(execution_time, 0.1))

        await memory_service.store_memory(
            content=f"DBT Model: {model_name}\n\nSQL:\n{model_sql}",
            memory_type="dbt_pattern",
            metadata={
                "model_name": model_name,
                "execution_time": execution_time,
                "test_pass_rate": test_pass_rate,
                "quality_score": quality_score,
                "test_results": test_results
            }
        )

    async def get_similar_patterns(self, request_description: str) -> List[Dict[str, Any]]:
        """Retrieve similar successful patterns from memory"""
        from adam.memory.project import ProjectAwareMemory

        memory_service = ProjectAwareMemory(self.project_id, "dbt_patterns")

        memories = await memory_service.search_memories(
            query=request_description,
            memory_types=["dbt_pattern"],
            limit=5
        )

        return [
            {
                "content": mem.content,
                "quality_score": mem.metadata.get('quality_score', 0),
                "model_name": mem.metadata.get('model_name', 'unknown')
            }
            for mem in memories
        ]


# ---------------------------------------------------------------------------
# Module-level singletons (convenience)
# ---------------------------------------------------------------------------

_dbt_knowledge_service: Optional[DBTKnowledgeService] = None


def get_dbt_knowledge_service() -> DBTKnowledgeService:
    """Get or create the global DBT knowledge service instance"""
    global _dbt_knowledge_service
    if _dbt_knowledge_service is None:
        _dbt_knowledge_service = DBTKnowledgeService()
    return _dbt_knowledge_service
