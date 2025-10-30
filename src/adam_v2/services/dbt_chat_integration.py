"""
DBT Chat Integration Service with Claude Haiku Router
Enables natural language interaction with DBT documentation features
Uses Claude Haiku 4.5 for intelligent intent detection
"""

import logging
import json
from typing import Dict, Optional, Tuple, List
from pathlib import Path

from .dbt_column_service import DBTColumnDocumentationService
from ...adam.llm.client import UnifiedLLMClient

logger = logging.getLogger(__name__)

class DBTChatIntegration:
    """
    Integrates DBT column documentation into ADAM's chat interface
    Uses Claude Haiku 4.5 for intelligent intent detection instead of keyword matching
    """

    def __init__(self):
        self.column_service = DBTColumnDocumentationService()
        self.llm_client = UnifiedLLMClient()

        # Intent categories
        self.valid_intents = [
            'document_model',      # Document columns in a specific model
            'document_all',        # Document all models
            'analyze_columns',     # Analyze columns in project
            'standardize',         # Standardize inconsistent columns
            'generate_schema',     # Generate schema.yml
            'suggest_improvements',# Get improvement suggestions
            'find_common',         # Find common columns
            'not_dbt'             # Not a DBT query
        ]

    async def is_dbt_query(self, message: str, workspace_path: Optional[str] = None) -> bool:
        """
        Use Claude Haiku to intelligently detect if this is a DBT-related query
        Fast and accurate - no keyword matching!
        """
        try:
            prompt = f"""Analyze if this user message is asking about DBT (data build tool) documentation, columns, or models.

User message: "{message}"

Context:
- DBT is a data transformation tool
- Users may ask about: documenting columns, analyzing models, generating schema files, checking column consistency
- They might not explicitly say "DBT" but ask about "columns", "models", "documentation", "schema.yml"

Response format (JSON):
{{
    "is_dbt_query": true/false,
    "confidence": 0.0-1.0,
    "reasoning": "brief explanation"
}}

Only respond with the JSON, nothing else."""

            response = await self.llm_client.complete(
                prompt=prompt,
                model="claude-haiku-4-5-20251001",  # Fast Haiku for routing
                temperature=0.1,  # Low temperature for consistent classification
                max_tokens=150
            )

            # Parse JSON response
            result = json.loads(response.content.strip())

            logger.info(f"DBT query detection - Confidence: {result['confidence']}, Is DBT: {result['is_dbt_query']}")

            return result['is_dbt_query'] and result['confidence'] > 0.6

        except Exception as e:
            logger.error(f"Error in DBT query detection: {str(e)}")
            # Fallback to simple keyword check
            dbt_keywords = ['dbt', 'column', 'schema.yml', 'document', 'model', 'table']
            return any(keyword in message.lower() for keyword in dbt_keywords)

    async def detect_intent_and_entities(self, message: str, active_file: Optional[str] = None) -> Dict:
        """
        Use Claude Haiku to detect user intent and extract entities
        Returns structured information about what the user wants
        """
        try:
            # Build context
            context_info = ""
            if active_file:
                file_name = Path(active_file).stem
                context_info = f"\nCurrently viewing file: {file_name}.sql"

            prompt = f"""Analyze this DBT-related user query and extract intent and entities.

User message: "{message}"{context_info}

Available intents:
- document_model: User wants to document columns in a model
- document_all: User wants to document all models in project
- analyze_columns: User wants to analyze column patterns/statistics
- standardize: User wants to standardize inconsistent column documentation
- generate_schema: User wants to generate schema.yml file
- suggest_improvements: User wants documentation improvement suggestions
- find_common: User wants to find common columns across models

Response format (JSON only):
{{
    "intent": "one_of_the_above",
    "confidence": 0.0-1.0,
    "entities": {{
        "model_name": "extracted model name or null",
        "scope": "current|all|specific",
        "use_ai": true,
        "preserve_existing": true
    }},
    "reasoning": "brief explanation"
}}"""

            response = await self.llm_client.complete(
                prompt=prompt,
                model="claude-haiku-4-5-20251001",
                temperature=0.2,
                max_tokens=200
            )

            # Parse JSON response
            result = json.loads(response.content.strip())

            # If model_name is null and active_file exists, use active file
            if not result['entities'].get('model_name') and active_file:
                result['entities']['model_name'] = Path(active_file).stem

            logger.info(f"Intent detected: {result['intent']} (confidence: {result['confidence']})")

            return result

        except Exception as e:
            logger.error(f"Error in intent detection: {str(e)}")
            # Fallback
            return {
                'intent': 'analyze_columns',
                'confidence': 0.5,
                'entities': {
                    'model_name': Path(active_file).stem if active_file else None,
                    'scope': 'current',
                    'use_ai': True,
                    'preserve_existing': True
                },
                'reasoning': 'Fallback due to parsing error'
            }

    async def handle_dbt_query(self, message: str, workspace_path: str,
                               active_file: Optional[str] = None) -> Dict:
        """
        Main entry point - handles DBT queries with intelligent routing
        """
        try:
            # First check if this is a DBT query
            if not await self.is_dbt_query(message, workspace_path):
                return {
                    'success': False,
                    'is_dbt': False,
                    'message': None
                }

            # Detect intent and extract entities
            detection = await self.detect_intent_and_entities(message, active_file)
            intent = detection['intent']
            entities = detection['entities']

            logger.info(f"Handling DBT query - Intent: {intent}, Entities: {entities}")

            # Route to appropriate handler
            handlers = {
                'document_model': self._handle_document_model,
                'document_all': self._handle_document_all,
                'analyze_columns': self._handle_analyze_columns,
                'standardize': self._handle_standardize,
                'generate_schema': self._handle_generate_schema,
                'suggest_improvements': self._handle_suggest_improvements,
                'find_common': self._handle_find_common,
            }

            handler = handlers.get(intent)
            if not handler:
                return {
                    'success': False,
                    'is_dbt': True,
                    'message': f"I detected a DBT query but I'm not sure how to handle it yet. Intent: {intent}"
                }

            # Execute handler
            result = await handler(workspace_path, entities, active_file)
            result['is_dbt'] = True
            result['intent'] = intent
            result['detection'] = detection

            return result

        except Exception as e:
            logger.error(f"Error handling DBT query: {str(e)}", exc_info=True)
            return {
                'success': False,
                'is_dbt': True,
                'message': f"Sorry, I encountered an error while processing your DBT request: {str(e)}"
            }

    async def _handle_document_model(self, workspace_path: str, entities: Dict,
                                    active_file: Optional[str]) -> Dict:
        """Document a specific model"""
        model_name = entities.get('model_name')

        if not model_name:
            return {
                'success': False,
                'message': "I need to know which model to document. Please specify a model name or open the model file in the editor."
            }

        result = await self.column_service.document_model_columns(
            project_path=workspace_path,
            model_name=model_name,
            use_ai=entities.get('use_ai', True),
            preserve_existing=entities.get('preserve_existing', True)
        )

        # Format response
        column_list = '\n'.join([
            f"- **{col['name']}**: {col['description']}" +
            (f" (FK → {col['references']})" if col.get('is_foreign_key') else "")
            for col in list(result['columns'].values())[:10]  # Show first 10
        ])

        more_cols = len(result['columns']) - 10
        if more_cols > 0:
            column_list += f"\n\n_...and {more_cols} more columns_"

        return {
            'success': True,
            'message': f"""## 📝 Column Documentation for `{model_name}`

I've analyzed and documented **{result['total_columns']} columns** using AI-powered pattern recognition.

### Sample Columns:
{column_list}

### Statistics:
- ✅ Documented: {result['documented_columns']}/{result['total_columns']}
- 🔍 Pattern-detected: {result['pattern_columns']}
- 🔗 Foreign keys: {result['foreign_keys']}

### Next Steps:
Would you like me to:
1. Save this to `schema.yml`
2. Show all columns
3. Generate tests for these columns""",
            'data': result,
            'actions': [
                {'label': 'Save to schema.yml', 'action': 'save_schema', 'data': result},
                {'label': 'Show all columns', 'action': 'show_all', 'data': result},
                {'label': 'Generate tests', 'action': 'generate_tests', 'data': result}
            ]
        }

    async def _handle_document_all(self, workspace_path: str, entities: Dict,
                                   active_file: Optional[str]) -> Dict:
        """Document all models in project"""
        return {
            'success': True,
            'message': """## 📚 Bulk Documentation

I'll document all models in your DBT project. This is a powerful feature that will:

1. Analyze all models in your project
2. Generate intelligent descriptions for every column
3. Detect patterns and relationships
4. Create comprehensive schema.yml files

⏱️ This may take a few minutes for large projects.

Would you like me to proceed? (Reply "yes" to start)""",
            'data': {'workspace_path': workspace_path},
            'requires_confirmation': True
        }

    async def _handle_analyze_columns(self, workspace_path: str, entities: Dict,
                                     active_file: Optional[str]) -> Dict:
        """Analyze all columns in project"""
        analysis = await self.column_service.analyze_columns(
            project_path=workspace_path,
            use_ai=True
        )

        # Format common columns
        common_cols = '\n'.join([
            f"- **{col['name']}**: Appears in {col['occurrence_count']} models "
            f"{'✅ Documented' if col['has_documentation'] else '⚠️ **Needs docs**'}"
            for col in analysis['common_columns'][:10]
        ])

        # Format patterns
        top_patterns = sorted(
            analysis['patterns_detected'].items(),
            key=lambda x: len(x[1]),
            reverse=True
        )[:5]

        patterns = '\n'.join([
            f"- **{pattern}**: {len(cols)} columns"
            for pattern, cols in top_patterns
            if len(cols) > 0
        ])

        # AI insights if available
        ai_section = ""
        if 'ai_insights' in analysis and analysis['ai_insights']:
            ai_section = f"""

### 🤖 AI Insights:
{chr(10).join(f'{i+1}. {insight}' for i, insight in enumerate(analysis['ai_insights']))}"""

        return {
            'success': True,
            'message': f"""## 📊 DBT Column Analysis Report

### Summary
- **Total Columns**: {analysis['summary']['total_columns']}
- **Documented**: {analysis['summary']['documented_columns']} ({analysis['summary']['documentation_coverage']})
- **Common Columns**: {analysis['summary']['common_columns']}
- **Need Attention**: {analysis['summary']['undocumented_common_columns']} ⚠️

### Top Common Columns
{common_cols}

### Detected Patterns
{patterns}{ai_section}

### 💡 Recommendations
1. **Priority**: Document the {analysis['summary']['undocumented_common_columns']} undocumented common columns
2. **Consistency**: Standardize descriptions for shared columns
3. **Testing**: Add relationship tests for foreign keys

What would you like me to help with?""",
            'data': analysis,
            'actions': [
                {'label': 'Document common columns', 'action': 'document_common'},
                {'label': 'Standardize inconsistent', 'action': 'standardize'},
                {'label': 'Show improvement suggestions', 'action': 'suggest_improvements'}
            ]
        }

    async def _handle_standardize(self, workspace_path: str, entities: Dict,
                                  active_file: Optional[str]) -> Dict:
        """Find columns needing standardization"""
        common = await self.column_service.find_common_columns(
            project_path=workspace_path,
            min_occurrences=2
        )

        needs_standardization = [c for c in common if c['needs_standardization']]

        if not needs_standardization:
            return {
                'success': True,
                'message': "🎉 **Great news!** All common columns have consistent documentation across your project."
            }

        column_list = '\n'.join([
            f"{i+1}. **{col['column_name']}**\n"
            f"   - Appears in: {col['occurrence_count']} models\n"
            f"   - Different descriptions: {len(col['existing_descriptions'])}\n"
            f"   - Suggested standard: _{col.get('suggested_standard_description', 'Analyzing...')}_"
            for i, col in enumerate(needs_standardization[:5])
        ])

        return {
            'success': True,
            'message': f"""## 🔄 Columns Needing Standardization

Found **{len(needs_standardization)} columns** with inconsistent documentation:

{column_list}

These columns would benefit from standardized descriptions to maintain consistency across your data warehouse.

Would you like me to standardize these automatically?""",
            'data': needs_standardization,
            'actions': [
                {'label': 'Standardize all', 'action': 'standardize_all'},
                {'label': 'Review one by one', 'action': 'review_standardization'}
            ]
        }

    async def _handle_generate_schema(self, workspace_path: str, entities: Dict,
                                     active_file: Optional[str]) -> Dict:
        """Generate schema.yml file"""
        yaml_content = await self.column_service.generate_schema_yml(
            project_path=workspace_path,
            model_path=active_file,
            use_ai=True,
            include_tests=True
        )

        # Get preview (first 20 lines)
        preview_lines = yaml_content.split('\n')[:20]
        preview = '\n'.join(preview_lines)
        if len(yaml_content.split('\n')) > 20:
            preview += '\n...'

        return {
            'success': True,
            'message': f"""## 📝 Generated `schema.yml`

I've created a comprehensive schema file with:
- ✅ Intelligent column descriptions (AI-powered)
- ✅ Suggested data quality tests
- ✅ Relationship tests for foreign keys
- ✅ Pattern-based test recommendations

### Preview:
```yaml
{preview}
```

Would you like me to save this file?""",
            'data': {'yaml_content': yaml_content, 'file_path': active_file},
            'actions': [
                {'label': 'Save file', 'action': 'save_yaml'},
                {'label': 'Show full content', 'action': 'show_full_yaml'},
                {'label': 'Modify before saving', 'action': 'edit_yaml'}
            ]
        }

    async def _handle_suggest_improvements(self, workspace_path: str, entities: Dict,
                                          active_file: Optional[str]) -> Dict:
        """Suggest documentation improvements"""
        improvements = await self.column_service.suggest_documentation_improvements(
            project_path=workspace_path,
            priority='high'
        )

        if not improvements:
            return {
                'success': True,
                'message': "🎉 **Excellent work!** Your documentation is in great shape. No high-priority improvements needed."
            }

        improvement_list = '\n'.join([
            f"**{i+1}. {imp['type'].replace('_', ' ').title()}**\n"
            f"   - Column: `{imp.get('column', 'N/A')}`\n"
            f"   - Impact: {imp['impact']}\n"
            f"   - Action: {imp['suggested_action']}\n"
            for i, imp in enumerate(improvements[:5])
        ])

        return {
            'success': True,
            'message': f"""## 💡 Documentation Improvement Suggestions

Found **{len(improvements)} high-priority improvements**:

{improvement_list}

These improvements will significantly enhance your documentation quality and data reliability.

Would you like me to apply any of these?""",
            'data': improvements,
            'actions': [
                {'label': 'Apply all', 'action': 'apply_all_improvements'},
                {'label': 'Apply selectively', 'action': 'apply_selective'},
                {'label': 'Show more details', 'action': 'show_improvement_details'}
            ]
        }

    async def _handle_find_common(self, workspace_path: str, entities: Dict,
                                  active_file: Optional[str]) -> Dict:
        """Find common columns across models"""
        common = await self.column_service.find_common_columns(
            project_path=workspace_path,
            min_occurrences=3
        )

        column_list = '\n'.join([
            f"- **{col['column_name']}**: {col['occurrence_count']} models "
            f"{'✅ Consistent' if not col['needs_standardization'] else '⚠️ **Needs standardization**'}"
            for col in common[:15]
        ])

        return {
            'success': True,
            'message': f"""## 🔍 Common Columns Across Models

Found **{len(common)} columns** appearing in 3+ models:

{column_list}

These are excellent candidates for:
1. Standardized documentation
2. Consistent data quality tests
3. Relationship validation

Would you like me to help standardize any of these?""",
            'data': common,
            'actions': [
                {'label': 'Standardize all common', 'action': 'standardize_common'},
                {'label': 'Review individually', 'action': 'review_common'}
            ]
        }

    def format_for_chat(self, response: Dict) -> str:
        """Format response for chat display"""
        if response.get('success'):
            return response['message']
        else:
            return f"⚠️ {response.get('message', 'An error occurred')}"