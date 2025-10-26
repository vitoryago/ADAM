"""
Column Intelligence Engine for DBT Documentation
Provides smart column documentation generation using AI and pattern recognition
"""

import re
import logging
from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
import json

from .parser import DBTManifestParser
from .direct_reader import DirectDBTReader

logger = logging.getLogger(__name__)

@dataclass
class ColumnPattern:
    """Represents a detected column pattern"""
    pattern: str
    pattern_type: str  # 'id', 'timestamp', 'flag', 'amount', 'count', 'email', 'phone', etc.
    standard_description: str
    confidence: float
    examples: List[str] = field(default_factory=list)

@dataclass
class ColumnInfo:
    """Information about a column across the project"""
    name: str
    appears_in_models: Set[str] = field(default_factory=set)
    data_types: Set[str] = field(default_factory=set)
    existing_descriptions: Dict[str, str] = field(default_factory=dict)
    transformations: List[str] = field(default_factory=list)
    pattern: Optional[ColumnPattern] = None
    suggested_description: Optional[str] = None
    is_foreign_key: bool = False
    references_table: Optional[str] = None
    references_column: Optional[str] = None

class ColumnIntelligenceEngine:
    """
    Intelligent column documentation engine that:
    1. Detects column patterns across all models
    2. Identifies common columns that need consistent documentation
    3. Generates intelligent descriptions using AI
    4. Learns from existing documentation patterns
    """

    def __init__(self, parser: Optional[DBTManifestParser] = None,
                 reader: Optional[DirectDBTReader] = None):
        self.parser = parser
        self.reader = reader
        self.column_catalog: Dict[str, ColumnInfo] = {}
        self.patterns = self._initialize_patterns()

    def _initialize_patterns(self) -> List[ColumnPattern]:
        """Initialize common column patterns with standard descriptions"""
        return [
            # IDs and Keys
            ColumnPattern(
                pattern=r"^.*_id$",
                pattern_type="id",
                standard_description="Unique identifier for {entity}",
                confidence=0.9,
                examples=["customer_id", "order_id", "product_id"]
            ),
            ColumnPattern(
                pattern=r"^.*_key$",
                pattern_type="key",
                standard_description="Key field for {entity}",
                confidence=0.85,
                examples=["primary_key", "foreign_key", "surrogate_key"]
            ),

            # Timestamps
            ColumnPattern(
                pattern=r"^.*_(at|timestamp)$",
                pattern_type="timestamp",
                standard_description="Timestamp when {action} occurred",
                confidence=0.95,
                examples=["created_at", "updated_at", "deleted_at"]
            ),
            ColumnPattern(
                pattern=r"^.*_date$",
                pattern_type="date",
                standard_description="Date of {event}",
                confidence=0.9,
                examples=["order_date", "birth_date", "start_date"]
            ),

            # Booleans
            ColumnPattern(
                pattern=r"^(is|has|was)_.*",
                pattern_type="boolean",
                standard_description="Boolean flag indicating if {condition}",
                confidence=0.95,
                examples=["is_active", "has_subscription", "was_deleted"]
            ),

            # Amounts and Money
            ColumnPattern(
                pattern=r"^.*_(amount|price|cost|fee|revenue|balance)$",
                pattern_type="money",
                standard_description="Monetary value representing {metric}",
                confidence=0.9,
                examples=["total_amount", "unit_price", "shipping_cost"]
            ),

            # Counts and Quantities
            ColumnPattern(
                pattern=r"^(num|count|qty|quantity)_.*",
                pattern_type="count",
                standard_description="Number of {items}",
                confidence=0.9,
                examples=["num_items", "count_orders", "qty_sold"]
            ),

            # Percentages and Ratios
            ColumnPattern(
                pattern=r"^.*_(pct|percent|ratio|rate)$",
                pattern_type="percentage",
                standard_description="Percentage/ratio of {metric}",
                confidence=0.85,
                examples=["conversion_rate", "discount_pct", "success_ratio"]
            ),

            # Status and State
            ColumnPattern(
                pattern=r"^.*_(status|state|stage)$",
                pattern_type="status",
                standard_description="Current status/state of {entity}",
                confidence=0.85,
                examples=["order_status", "payment_state", "workflow_stage"]
            ),

            # Names and Descriptions
            ColumnPattern(
                pattern=r"^.*_(name|title|description)$",
                pattern_type="text",
                standard_description="Name/description of {entity}",
                confidence=0.8,
                examples=["product_name", "category_title", "item_description"]
            ),

            # Contact Information
            ColumnPattern(
                pattern=r"^.*_email$",
                pattern_type="email",
                standard_description="Email address for {entity}",
                confidence=0.95,
                examples=["customer_email", "user_email"]
            ),
            ColumnPattern(
                pattern=r"^.*_phone$",
                pattern_type="phone",
                standard_description="Phone number for {entity}",
                confidence=0.95,
                examples=["contact_phone", "mobile_phone"]
            ),

            # Geographic
            ColumnPattern(
                pattern=r"^.*(country|city|state|zip|postal).*$",
                pattern_type="geographic",
                standard_description="Geographic location - {location_type}",
                confidence=0.85,
                examples=["billing_country", "shipping_city", "postal_code"]
            ),

            # Segment/Analytics specific
            ColumnPattern(
                pattern=r"^(event|user|anonymous)_id$",
                pattern_type="segment_id",
                standard_description="Segment tracking identifier",
                confidence=0.95,
                examples=["event_id", "user_id", "anonymous_id"]
            ),
            ColumnPattern(
                pattern=r"^(utm_|gclid|fbclid).*",
                pattern_type="marketing_param",
                standard_description="Marketing attribution parameter",
                confidence=0.9,
                examples=["utm_source", "utm_campaign", "gclid"]
            ),
        ]

    def analyze_project_columns(self) -> Dict[str, ColumnInfo]:
        """
        Analyze all columns across the entire DBT project
        Returns a catalog of all columns with their usage patterns
        """
        logger.info("Starting project-wide column analysis")

        if self.parser and self.parser.manifest:
            # Use manifest if available
            self._analyze_from_manifest()
        elif self.reader:
            # Use direct reader if no manifest
            self._analyze_from_files()
        else:
            raise ValueError("Either parser or reader must be provided")

        # Detect patterns for all columns
        self._detect_column_patterns()

        # Identify foreign keys
        self._identify_foreign_keys()

        logger.info(f"Analyzed {len(self.column_catalog)} unique columns")
        return self.column_catalog

    def _analyze_from_manifest(self):
        """Analyze columns from parsed manifest"""
        for model_name, model in self.parser.manifest.nodes.items():
            if not model_name.startswith("model."):
                continue

            model_short_name = model_name.split(".")[-1]

            # Get columns from model
            if hasattr(model, 'columns'):
                for col_name, col_info in model.columns.items():
                    if col_name not in self.column_catalog:
                        self.column_catalog[col_name] = ColumnInfo(name=col_name)

                    self.column_catalog[col_name].appears_in_models.add(model_short_name)

                    if hasattr(col_info, 'data_type') and col_info.data_type:
                        self.column_catalog[col_name].data_types.add(col_info.data_type)

                    if hasattr(col_info, 'description') and col_info.description:
                        self.column_catalog[col_name].existing_descriptions[model_short_name] = col_info.description

    def _analyze_from_files(self):
        """Analyze columns from direct file reading"""
        models = self.reader.get_all_models()

        for model_name, model_info in models.items():
            # Parse SQL to extract columns (simplified - production would use sqlglot)
            columns = self._extract_columns_from_sql(model_info['sql'])

            for col_name in columns:
                if col_name not in self.column_catalog:
                    self.column_catalog[col_name] = ColumnInfo(name=col_name)

                self.column_catalog[col_name].appears_in_models.add(model_name)

                # Get description from schema.yml if exists
                if 'columns' in model_info and col_name in model_info['columns']:
                    desc = model_info['columns'][col_name].get('description')
                    if desc:
                        self.column_catalog[col_name].existing_descriptions[model_name] = desc

    def _extract_columns_from_sql(self, sql: str) -> List[str]:
        """
        Extract column names from SQL (simplified version)
        Production should use sqlglot for accurate parsing
        """
        columns = []

        # Extract from SELECT clause
        select_pattern = r'SELECT\s+(.*?)\s+FROM'
        select_match = re.search(select_pattern, sql, re.IGNORECASE | re.DOTALL)

        if select_match:
            select_clause = select_match.group(1)
            # Split by comma and extract column names
            for part in select_clause.split(','):
                # Handle aliases
                if ' as ' in part.lower():
                    col_name = part.lower().split(' as ')[-1].strip()
                else:
                    col_name = part.strip().split('.')[-1]

                # Clean up
                col_name = col_name.strip('`"[]')
                if col_name and col_name != '*':
                    columns.append(col_name)

        return columns

    def _detect_column_patterns(self):
        """Detect patterns for all columns in the catalog"""
        for col_name, col_info in self.column_catalog.items():
            for pattern in self.patterns:
                if re.match(pattern.pattern, col_name, re.IGNORECASE):
                    col_info.pattern = pattern
                    break

    def _identify_foreign_keys(self):
        """Identify potential foreign key relationships"""
        for col_name, col_info in self.column_catalog.items():
            if col_name.endswith('_id') and col_name != 'id':
                # Potential foreign key
                col_info.is_foreign_key = True

                # Try to identify the referenced table
                table_name = col_name[:-3]  # Remove '_id'
                potential_tables = [
                    f"dim_{table_name}",
                    f"fct_{table_name}",
                    f"stg_{table_name}",
                    table_name + "s",  # Plural
                    table_name
                ]

                # Check if any of these tables exist
                all_models = set()
                if self.parser:
                    all_models = set(m.split('.')[-1] for m in self.parser.manifest.nodes.keys())
                elif self.reader:
                    all_models = set(self.reader.get_all_models().keys())

                for table in potential_tables:
                    if table in all_models:
                        col_info.references_table = table
                        col_info.references_column = "id"  # Assume 'id' column
                        break

    def find_common_columns(self, min_occurrences: int = 2) -> List[Tuple[str, ColumnInfo]]:
        """
        Find columns that appear in multiple models
        These are candidates for standardized documentation
        """
        common_columns = [
            (name, info) for name, info in self.column_catalog.items()
            if len(info.appears_in_models) >= min_occurrences
        ]

        # Sort by frequency of occurrence
        common_columns.sort(key=lambda x: len(x[1].appears_in_models), reverse=True)

        return common_columns

    def suggest_column_description(self, column_name: str,
                                  model_name: Optional[str] = None,
                                  context: Optional[str] = None) -> str:
        """
        Generate an intelligent description for a column
        Uses pattern matching and context to create meaningful descriptions
        """
        col_info = self.column_catalog.get(column_name, ColumnInfo(name=column_name))

        # Check if we have a pattern match
        if col_info.pattern:
            description = col_info.pattern.standard_description

            # Replace placeholders with context
            if '{entity}' in description:
                entity = self._extract_entity_from_name(column_name, model_name)
                description = description.replace('{entity}', entity)

            if '{action}' in description:
                action = self._extract_action_from_name(column_name)
                description = description.replace('{action}', action)

            if '{event}' in description:
                event = self._extract_event_from_name(column_name)
                description = description.replace('{event}', event)

            if '{metric}' in description:
                metric = self._extract_metric_from_name(column_name)
                description = description.replace('{metric}', metric)

            if '{condition}' in description:
                condition = self._extract_condition_from_name(column_name)
                description = description.replace('{condition}', condition)

            if '{items}' in description:
                items = self._extract_items_from_name(column_name)
                description = description.replace('{items}', items)

            if '{location_type}' in description:
                location = self._extract_location_type(column_name)
                description = description.replace('{location_type}', location)

            # Add foreign key information if applicable
            if col_info.is_foreign_key and col_info.references_table:
                description += f". References {col_info.references_table}.{col_info.references_column or 'id'}"

            # Add context if provided
            if context:
                description += f". {context}"

            return description

        # If no pattern matches, create a generic description
        return f"Column {column_name} in {model_name or 'model'}"

    def _extract_entity_from_name(self, column_name: str, model_name: Optional[str]) -> str:
        """Extract the entity name from column name"""
        # Remove common suffixes
        entity = re.sub(r'_(id|key|name|status|date|at|amount|count)$', '', column_name)

        # Use model name as context if available
        if model_name:
            if 'customer' in model_name.lower():
                return 'customer'
            elif 'order' in model_name.lower():
                return 'order'
            elif 'product' in model_name.lower():
                return 'product'

        return entity.replace('_', ' ')

    def _extract_action_from_name(self, column_name: str) -> str:
        """Extract action from timestamp column names"""
        if 'created' in column_name:
            return 'record was created'
        elif 'updated' in column_name:
            return 'record was last updated'
        elif 'deleted' in column_name:
            return 'record was deleted'
        elif 'modified' in column_name:
            return 'record was modified'
        elif 'processed' in column_name:
            return 'record was processed'
        else:
            action = column_name.replace('_at', '').replace('_timestamp', '')
            return f"{action.replace('_', ' ')} occurred"

    def _extract_event_from_name(self, column_name: str) -> str:
        """Extract event from date column names"""
        event = column_name.replace('_date', '')
        return event.replace('_', ' ')

    def _extract_metric_from_name(self, column_name: str) -> str:
        """Extract metric from amount/price column names"""
        metric = re.sub(r'_(amount|price|cost|fee|revenue|balance)$', '', column_name)
        return metric.replace('_', ' ')

    def _extract_condition_from_name(self, column_name: str) -> str:
        """Extract condition from boolean column names"""
        condition = re.sub(r'^(is|has|was)_', '', column_name)
        return condition.replace('_', ' ')

    def _extract_items_from_name(self, column_name: str) -> str:
        """Extract items from count column names"""
        items = re.sub(r'^(num|count|qty|quantity)_', '', column_name)
        return items.replace('_', ' ')

    def _extract_location_type(self, column_name: str) -> str:
        """Extract location type from geographic columns"""
        if 'country' in column_name:
            return 'country'
        elif 'city' in column_name:
            return 'city'
        elif 'state' in column_name:
            return 'state or province'
        elif 'zip' in column_name or 'postal' in column_name:
            return 'postal/ZIP code'
        else:
            return 'geographic location'

    def generate_documentation_report(self) -> Dict:
        """
        Generate a comprehensive report on column documentation status
        """
        total_columns = len(self.column_catalog)
        documented_columns = sum(
            1 for col in self.column_catalog.values()
            if col.existing_descriptions
        )

        common_columns = self.find_common_columns()
        undocumented_common = [
            (name, info) for name, info in common_columns
            if not info.existing_descriptions
        ]

        return {
            'summary': {
                'total_columns': total_columns,
                'documented_columns': documented_columns,
                'documentation_coverage': f"{(documented_columns / total_columns * 100):.1f}%" if total_columns > 0 else "0%",
                'common_columns': len(common_columns),
                'undocumented_common_columns': len(undocumented_common)
            },
            'common_columns': [
                {
                    'name': name,
                    'appears_in': list(info.appears_in_models),
                    'occurrence_count': len(info.appears_in_models),
                    'has_documentation': bool(info.existing_descriptions),
                    'pattern_detected': info.pattern.pattern_type if info.pattern else None,
                    'suggested_description': self.suggest_column_description(name)
                }
                for name, info in common_columns[:20]  # Top 20 most common
            ],
            'patterns_detected': {
                pattern_type: [
                    col_name for col_name, col_info in self.column_catalog.items()
                    if col_info.pattern and col_info.pattern.pattern_type == pattern_type
                ]
                for pattern_type in set(p.pattern_type for p in self.patterns)
            },
            'foreign_keys': [
                {
                    'column': name,
                    'references': f"{info.references_table}.{info.references_column}"
                    if info.references_table else "Unknown",
                    'appears_in': list(info.appears_in_models)
                }
                for name, info in self.column_catalog.items()
                if info.is_foreign_key
            ]
        }

    def export_suggestions_to_yaml(self) -> Dict:
        """
        Export column documentation suggestions in YAML format
        Ready to be added to schema.yml files
        """
        suggestions = {}

        for col_name, col_info in self.column_catalog.items():
            if col_info.appears_in_models:
                # Group by model
                for model in col_info.appears_in_models:
                    if model not in suggestions:
                        suggestions[model] = {
                            'name': model,
                            'columns': []
                        }

                    # Check if column already has description
                    existing_desc = col_info.existing_descriptions.get(model)

                    column_def = {
                        'name': col_name,
                        'description': existing_desc or self.suggest_column_description(col_name, model)
                    }

                    # Add tests based on pattern
                    tests = []
                    if col_info.pattern:
                        if col_info.pattern.pattern_type == 'id':
                            tests.extend(['unique', 'not_null'])
                        elif col_info.pattern.pattern_type == 'boolean':
                            tests.append('not_null')
                        elif col_info.pattern.pattern_type in ['timestamp', 'date']:
                            tests.append('not_null')

                    if col_info.is_foreign_key and col_info.references_table:
                        tests.append({
                            'relationships': {
                                'to': f"ref('{col_info.references_table}')",
                                'field': col_info.references_column or 'id'
                            }
                        })

                    if tests:
                        column_def['tests'] = tests

                    suggestions[model]['columns'].append(column_def)

        return suggestions