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
        self.knowledge_base_path = Path(__file__).parent.parent.parent.parent
        self.patterns_file = self.knowledge_base_path / "dbt_patterns.yaml"
        self.knowledge_file = self.knowledge_base_path / "DBT_KNOWLEDGE.md"
        
        # Load patterns and knowledge
        self.patterns = self._load_patterns()
        self.knowledge = self._load_knowledge()
        
        # Pattern matchers for detecting DBT needs
        self.dbt_indicators = [
            'dbt', 'incremental', 'macro', 'jinja',
            'materialized', 'ref(', 'source(', 'snapshot',
            'staging', 'intermediate', 'mart', 'fact', 'dimension',
            'pdt', 'looker', 'derived_table'
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
        """Detect if a query needs DBT knowledge"""
        query_lower = query.lower()
        return any(indicator in query_lower for indicator in self.dbt_indicators)
    
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
        
        # Build context string
        context_parts = []
        
        if knowledge['detected_need']:
            context_parts.append(f"DBT Context Detected: {', '.join(knowledge['detected_need'])}")
        
        if knowledge['best_practices']:
            context_parts.append("\nRelevant Best Practices:")
            context_parts.extend(knowledge['best_practices'][:1])  # Add first best practice
        
        if knowledge['patterns']:
            context_parts.append("\nAvailable Patterns:")
            for pattern_type, pattern_data in list(knowledge['patterns'].items())[:2]:
                context_parts.append(f"- {pattern_type}: {pattern_data.get('description', 'Available')}")
        
        if context_parts:
            context = "\n".join(context_parts)
            return f"{query}\n\n[DBT Knowledge Base Context]\n{context}"
        
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