"""
DBT Knowledge Service for ADAM v2
Provides intelligent DBT pattern retrieval and template generation
"""

import yaml
import os
from pathlib import Path
from typing import Dict, List, Optional, Any
import re
import logging
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

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

class DBTKnowledgeService:
    """Service for retrieving and applying DBT best practices"""
    
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
    
    def detect_dbt_context(self, query: str) -> bool:
        """Detect if a query needs DBT knowledge using simple keyword check (fallback)"""
        query_lower = query.lower()
        return any(indicator in query_lower for indicator in self.dbt_indicators)
    
    async def intelligent_dbt_detection(self, query: str) -> tuple[bool, float]:
        """Use LLM to intelligently detect if query needs DBT knowledge"""
        try:
            # Use Claude Haiku for fast, accurate detection
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
                model="gpt-4o-mini",  # Fast and accurate
                temperature=0.1,  # Low temperature for consistency
                max_tokens=100
            )
            
            # Parse response
            import json
            content = response.content.strip()
            
            # Clean JSON response
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()
            
            # Find JSON object
            if not content.startswith('{'):
                json_start = content.find('{')
                if json_start != -1:
                    content = content[json_start:]
            
            # Find matching closing brace
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
            
            logger.info(f"DBT detection: {result['needs_dbt']} (confidence: {result['confidence']:.2f}) - {result['reason']}")
            return result['needs_dbt'], result['confidence']
            
        except Exception as e:
            logger.warning(f"Intelligent DBT detection failed, falling back to keywords: {e}")
            # Fallback to keyword detection
            return self.detect_dbt_context(query), 0.5
    
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
        """Retrieve a specific macro template"""
        macros = self.patterns.get('macros', {})
        macro = macros.get(macro_name, {})
        return macro.get('template') if macro else None
    
    def get_test_pattern(self, test_type: str) -> Optional[str]:
        """Retrieve a test pattern"""
        test_patterns = self.patterns.get('test_patterns', {})
        if test_type in test_patterns.get('basic_tests', []):
            return test_type  # Basic test name
        
        quality_tests = test_patterns.get('data_quality_tests', {})
        if test_type in quality_tests:
            return quality_tests[test_type].get('template')
        
        return None
    
    def get_incremental_strategy(self, strategy_name: str) -> Dict[str, Any]:
        """Get incremental materialization strategy"""
        strategies = self.patterns.get('incremental_strategies', {})
        return strategies.get(strategy_name, strategies.get('merge_on_unique_key', {}))
    
    def generate_model_template(self, request: DBTModelRequest) -> str:
        """Generate a complete DBT model based on request"""
        # Get layer-specific pattern
        layer_pattern = self.get_layer_pattern(request.layer)
        
        if not layer_pattern:
            return f"-- No pattern found for layer: {request.layer}"
        
        # Get the appropriate template
        sql_template = layer_pattern.get('sql_template', '')
        config_template = layer_pattern.get('config_template', '')
        
        # Build config block
        config = self._build_config(request, layer_pattern)
        
        # Build SQL
        sql = self._build_sql(request, sql_template)
        
        # Combine config and SQL
        model = f"{config}\n\n{sql}"
        
        # Add documentation header
        description = self._build_description(request, layer_pattern)
        model = f"-- {description}\n\n{model}"
        
        return model
    
    def _build_config(self, request: DBTModelRequest, pattern: Dict) -> str:
        """Build model configuration block"""
        config_template = pattern.get('config_template', '')
        
        # Determine materialization
        if request.layer == ModelLayer.STAGING:
            materialization = 'view'
        elif request.layer == ModelLayer.INTERMEDIATE:
            materialization = 'ephemeral'
        elif request.layer == ModelLayer.MART_FACT and request.unique_key:
            materialization = 'incremental'
        else:
            materialization = 'table'
        
        # Format cluster keys
        cluster_keys = request.cluster_keys or ['id']
        cluster_keys_str = str(cluster_keys).replace("'", '"')
        
        # Format merge columns for incremental
        merge_columns = "['updated_at', 'status']" if materialization == 'incremental' else "[]"
        
        # Replace placeholders
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
        """Build SQL query from template"""
        # This is a simplified version - in production, would need more sophisticated templating
        
        if request.layer == ModelLayer.STAGING:
            # Build staging SQL
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
            # Build intermediate SQL
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
            # Build fact table SQL
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
        """Build model description"""
        description_template = pattern.get('description_template', '')
        
        if not description_template:
            return f"Model: {request.model_name}"
        
        # Simple template replacement
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
    
    def get_relevant_knowledge(self, query: str) -> Dict[str, Any]:
        """Get all relevant DBT knowledge for a query"""
        
        # Detect what type of DBT help is needed
        query_lower = query.lower()
        
        knowledge_response = {
            'detected_need': [],
            'patterns': {},
            'macros': [],
            'best_practices': [],
            'examples': []
        }
        
        # Detect needs
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
        
        # Add relevant best practices from knowledge
        if knowledge_response['detected_need']:
            sections = self._extract_relevant_sections(
                self.knowledge, 
                knowledge_response['detected_need']
            )
            knowledge_response['best_practices'] = sections
        
        return knowledge_response
    
    def _extract_relevant_sections(self, knowledge: str, needs: List[str]) -> List[str]:
        """Extract relevant sections from knowledge markdown"""
        sections = []
        
        # Simple section extraction based on headers
        lines = knowledge.split('\n')
        current_section = []
        capturing = False
        
        for line in lines:
            # Check if this line starts a relevant section
            if line.startswith('#'):
                # Check if we should capture this section
                section_relevant = any(need in line.lower() for need in needs)
                if section_relevant:
                    capturing = True
                    if current_section:
                        sections.append('\n'.join(current_section))
                    current_section = [line]
                elif capturing and line.count('#') <= 2:  # New major section
                    capturing = False
                    if current_section:
                        sections.append('\n'.join(current_section))
                    current_section = []
            elif capturing:
                current_section.append(line)
        
        # Add last section if capturing
        if capturing and current_section:
            sections.append('\n'.join(current_section))
        
        return sections[:3]  # Return top 3 most relevant sections
    
    def suggest_improvements(self, model_sql: str) -> List[str]:
        """Analyze a DBT model and suggest improvements"""
        suggestions = []
        
        # Check for common anti-patterns
        if 'select *' in model_sql.lower() and 'staging' not in model_sql.lower():
            suggestions.append("Avoid SELECT * in non-staging models. Explicitly list columns for clarity.")
        
        if not '{{' in model_sql:
            suggestions.append("Consider using DBT macros for reusable logic (ref(), source(), etc.)")
        
        if 'join' in model_sql.lower() and 'staging' in model_sql.lower():
            suggestions.append("Complex joins should be in intermediate layer, not staging")
        
        if 'incremental' in model_sql and 'unique_key' not in model_sql:
            suggestions.append("Incremental models should specify a unique_key in config")
        
        if len(model_sql.split('\n')) > 200:
            suggestions.append("Consider breaking this large model into smaller intermediate models")
        
        return suggestions
    
    def validate_model_name(self, name: str, layer: ModelLayer) -> bool:
        """Validate model name follows conventions"""
        prefixes = {
            ModelLayer.STAGING: 'stg_',
            ModelLayer.INTERMEDIATE: 'int_',
            ModelLayer.MART_FACT: 'fct_',
            ModelLayer.MART_DIMENSION: 'dim_'
        }
        
        expected_prefix = prefixes.get(layer, '')
        return name.startswith(expected_prefix) if expected_prefix else True
    
    def enhance_query_with_dbt_context(self, query: str) -> str:
        """Enhance a query with relevant DBT context"""
        if not self.detect_dbt_context(query):
            return query
        
        # Get relevant knowledge
        knowledge = self.get_relevant_knowledge(query)
        
        # Build comprehensive context string
        context_parts = []
        
        # Add a comprehensive summary from our knowledge base
        context_parts.append("[DBT Knowledge Base - Snowflake Optimized]")
        context_parts.append("=" * 50)
        
        # Core principles
        context_parts.append("\n📋 DBT Best Practices (from your knowledge base):")
        context_parts.append("• Layer Architecture: Staging → Intermediate → Marts")
        context_parts.append("• Naming: stg_<source>__<entity>, int_<entity>__<transform>, fct_/dim_")
        context_parts.append("• Staging: Views, light transforms only (rename, cast, filter)")
        context_parts.append("• Intermediate: Ephemeral, business logic, cross-source joins")
        context_parts.append("• Marts: Tables/Incremental, business-ready, clustered")
        
        # Snowflake specifics
        context_parts.append("\n❄️ Snowflake Optimizations:")
        context_parts.append("• Cluster large tables (>1GB) by [date_column, high_cardinality_id]")
        context_parts.append("• Use transient=true for staging (no Time Travel needed)")
        context_parts.append("• Set AUTO_CLUSTERING = TRUE for fact tables")
        context_parts.append("• Warehouse sizing: Use XL for heavy transforms")
        
        # If incremental detected, add incremental patterns
        if 'incremental' in knowledge['detected_need']:
            context_parts.append("\n🔄 Incremental Patterns:")
            if 'incremental' in knowledge['patterns']:
                strategies = knowledge['patterns']['incremental']
                for strategy_name, strategy_data in list(strategies.items())[:2]:
                    context_parts.append(f"• {strategy_name}: {strategy_data.get('description', '')}")
            context_parts.append("• Always add lookback safety (3 days)")
            context_parts.append("• Use merge strategy for updates, append for immutable events")
        
        # If testing detected, add test patterns
        if 'test' in query.lower():
            context_parts.append("\n🧪 Testing Strategy:")
            context_parts.append("• Schema tests: unique, not_null, relationships")
            context_parts.append("• Custom tests: volume anomalies, freshness checks")
            context_parts.append("• Severity: error (blocks deploy), warn (alerts)")
        
        # Add relevant macros if detected
        if 'macro' in query.lower() and knowledge['macros']:
            context_parts.append(f"\n🔧 Available Macros: {', '.join(knowledge['macros'][:5])}")
        
        # Include actual best practice sections if available
        if knowledge['best_practices']:
            context_parts.append("\n📚 Detailed Best Practices:")
            # Include more substantial content
            for section in knowledge['best_practices'][:2]:  # Include 2 sections
                # Limit each section to avoid token overflow
                section_lines = section.split('\n')[:20]  # First 20 lines
                context_parts.append('\n'.join(section_lines))
        
        # Add example if relevant pattern exists
        if knowledge['patterns']:
            context_parts.append("\n💡 Remember:")
            context_parts.append("• One source = one staging model")
            context_parts.append("• Test everything (100% coverage on keys)")
            context_parts.append("• Document as you build")
            context_parts.append("• Use ref() and source() always")
        
        if context_parts:
            context = "\n".join(context_parts)
            enhanced = f"{query}\n\n{context}"
            
            # Log size of enhancement
            logger.info(f"Enhanced query from {len(query)} to {len(enhanced)} chars")
            
            return enhanced
        
        return query


# Integration with ADAM's memory service for self-learning
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
        from services.memory_service import ProjectMemoryService, MemoryType
        
        memory_service = ProjectMemoryService(self.project_id, "dbt_patterns")
        
        # Calculate quality score
        test_pass_rate = sum(test_results.values()) / len(test_results) if test_results else 0
        quality_score = test_pass_rate * (1 / max(execution_time, 0.1))  # Better if faster and passes tests
        
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
        from services.memory_service import ProjectMemoryService
        
        memory_service = ProjectMemoryService(self.project_id, "dbt_patterns")
        
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