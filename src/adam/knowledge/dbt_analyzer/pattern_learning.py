"""
Pattern Learning System
Learns from existing documentation to create project-specific patterns
Adapts to YOUR naming conventions and business domain
"""

import logging
import re
from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, field
from collections import defaultdict, Counter
import json

logger = logging.getLogger(__name__)

@dataclass
class LearnedPattern:
    """A pattern learned from existing documentation"""
    pattern_regex: str
    pattern_name: str
    example_columns: List[str]
    learned_descriptions: List[str]
    confidence_score: float
    occurrences: int
    standard_description_template: str

@dataclass
class BusinessTerm:
    """A business term identified in the project"""
    term: str
    definition: str
    related_columns: List[str]
    confidence: float
    examples: List[str] = field(default_factory=list)

class PatternLearningEngine:
    """
    Learns patterns from existing documentation in the project
    Creates a custom knowledge base specific to YOUR business
    """

    def __init__(self):
        self.learned_patterns: List[LearnedPattern] = []
        self.business_glossary: Dict[str, BusinessTerm] = {}
        self.naming_conventions: Dict[str, List[str]] = defaultdict(list)
        self.description_templates: Dict[str, str] = {}

    def learn_from_project(self, column_catalog: Dict) -> Dict:
        """
        Analyze existing documentation to learn patterns
        column_catalog: {column_name: ColumnInfo}
        """
        logger.info("Learning patterns from existing documentation...")

        # Step 1: Learn naming conventions
        self._learn_naming_conventions(column_catalog)

        # Step 2: Extract business terms
        self._extract_business_terms(column_catalog)

        # Step 3: Learn column patterns
        self._learn_column_patterns(column_catalog)

        # Step 4: Build description templates
        self._build_description_templates(column_catalog)

        return {
            'learned_patterns': len(self.learned_patterns),
            'business_terms': len(self.business_glossary),
            'naming_conventions': {k: len(v) for k, v in self.naming_conventions.items()},
            'templates': len(self.description_templates)
        }

    def _learn_naming_conventions(self, column_catalog: Dict):
        """
        Learn project-specific naming conventions
        E.g., "evt_" prefix for events, "_key" suffix for foreign keys, etc.
        """
        logger.info("Learning naming conventions...")

        # Analyze prefixes
        prefix_patterns = defaultdict(list)
        suffix_patterns = defaultdict(list)

        for col_name in column_catalog.keys():
            # Extract prefixes (first 3-4 chars before underscore)
            prefix_match = re.match(r'^([a-z]{2,4})_', col_name)
            if prefix_match:
                prefix = prefix_match.group(1)
                prefix_patterns[prefix].append(col_name)

            # Extract suffixes (last part after underscore)
            suffix_match = re.search(r'_([a-z]{2,10})$', col_name)
            if suffix_match:
                suffix = suffix_match.group(1)
                suffix_patterns[suffix].append(col_name)

        # Keep only significant patterns (appear 3+ times)
        self.naming_conventions['prefixes'] = {
            prefix: cols for prefix, cols in prefix_patterns.items()
            if len(cols) >= 3
        }

        self.naming_conventions['suffixes'] = {
            suffix: cols for suffix, cols in suffix_patterns.items()
            if len(cols) >= 3
        }

        logger.info(f"Found {len(self.naming_conventions['prefixes'])} prefix patterns")
        logger.info(f"Found {len(self.naming_conventions['suffixes'])} suffix patterns")

    def _extract_business_terms(self, column_catalog: Dict):
        """
        Extract business-specific terminology from existing descriptions
        E.g., "MRR", "ARR", "Churn", "LTV", etc.
        """
        logger.info("Extracting business terms...")

        # Collect all descriptions
        all_descriptions = []
        term_to_columns = defaultdict(list)

        for col_name, col_info in column_catalog.items():
            if hasattr(col_info, 'existing_descriptions'):
                for desc in col_info.existing_descriptions.values():
                    if desc:
                        all_descriptions.append((col_name, desc))

        # Extract potential business terms (capitalized words, acronyms)
        for col_name, desc in all_descriptions:
            # Find acronyms (2-5 uppercase letters)
            acronyms = re.findall(r'\b([A-Z]{2,5})\b', desc)
            for acronym in acronyms:
                term_to_columns[acronym].append((col_name, desc))

            # Find capitalized terms
            cap_terms = re.findall(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b', desc)
            for term in cap_terms:
                if term not in ['The', 'A', 'An', 'This', 'That']:
                    term_to_columns[term].append((col_name, desc))

        # Create business terms (appear in 2+ columns)
        for term, occurrences in term_to_columns.items():
            if len(occurrences) >= 2:
                # Extract definition from first occurrence
                sample_desc = occurrences[0][1]

                # Try to extract context around the term
                pattern = rf'\b{re.escape(term)}\b[^.]*\.?'
                match = re.search(pattern, sample_desc)
                definition = match.group(0) if match else sample_desc

                self.business_glossary[term] = BusinessTerm(
                    term=term,
                    definition=definition,
                    related_columns=[col for col, _ in occurrences],
                    confidence=min(len(occurrences) / 10.0, 1.0),
                    examples=[desc for _, desc in occurrences[:3]]
                )

        logger.info(f"Extracted {len(self.business_glossary)} business terms")

    def _learn_column_patterns(self, column_catalog: Dict):
        """
        Learn column patterns from existing documentation
        Groups similar columns and extracts common description patterns
        """
        logger.info("Learning column patterns...")

        # Group columns by similar names
        pattern_groups = defaultdict(list)

        for col_name, col_info in column_catalog.items():
            if not hasattr(col_info, 'existing_descriptions') or not col_info.existing_descriptions:
                continue

            # Extract base pattern (remove specific identifiers)
            base_pattern = self._extract_base_pattern(col_name)

            if base_pattern:
                pattern_groups[base_pattern].append((col_name, col_info))

        # Analyze each pattern group
        for pattern_key, columns in pattern_groups.items():
            if len(columns) >= 3:  # Need at least 3 examples
                descriptions = []
                example_cols = []

                for col_name, col_info in columns:
                    example_cols.append(col_name)
                    if col_info.existing_descriptions:
                        descriptions.extend(col_info.existing_descriptions.values())

                if descriptions:
                    # Find common words in descriptions
                    template = self._extract_description_template(descriptions)

                    # Create regex pattern
                    regex_pattern = self._create_regex_from_pattern(pattern_key)

                    learned_pattern = LearnedPattern(
                        pattern_regex=regex_pattern,
                        pattern_name=pattern_key,
                        example_columns=example_cols[:5],
                        learned_descriptions=descriptions[:5],
                        confidence_score=min(len(columns) / 10.0, 1.0),
                        occurrences=len(columns),
                        standard_description_template=template
                    )

                    self.learned_patterns.append(learned_pattern)

        logger.info(f"Learned {len(self.learned_patterns)} column patterns")

    def _extract_base_pattern(self, column_name: str) -> Optional[str]:
        """
        Extract a base pattern from column name
        E.g., "customer_id" -> "entity_id", "order_created_at" -> "entity_created_at"
        """
        # Common entity-agnostic patterns
        patterns = [
            (r'^(.+)_id$', 'entity_id'),
            (r'^(.+)_key$', 'entity_key'),
            (r'^(.+)_at$', 'entity_at'),
            (r'^(.+)_date$', 'entity_date'),
            (r'^(.+)_count$', 'entity_count'),
            (r'^(.+)_amount$', 'entity_amount'),
            (r'^(.+)_price$', 'entity_price'),
            (r'^(.+)_status$', 'entity_status'),
            (r'^(.+)_name$', 'entity_name'),
            (r'^is_(.+)$', 'is_entity'),
            (r'^has_(.+)$', 'has_entity'),
            (r'^num_(.+)$', 'num_entity'),
            (r'^total_(.+)$', 'total_entity'),
        ]

        for pattern, template in patterns:
            if re.match(pattern, column_name):
                return template

        return None

    def _extract_description_template(self, descriptions: List[str]) -> str:
        """
        Extract a common template from multiple descriptions
        """
        if not descriptions:
            return ""

        # Find most common words (excluding stop words)
        stop_words = {'the', 'a', 'an', 'of', 'for', 'to', 'in', 'on', 'at', 'by', 'with'}

        all_words = []
        for desc in descriptions:
            words = desc.lower().split()
            all_words.extend([w for w in words if w not in stop_words and len(w) > 2])

        # Count word frequency
        word_counts = Counter(all_words)

        # Find most common words (appear in 30%+ of descriptions)
        threshold = len(descriptions) * 0.3
        common_words = [word for word, count in word_counts.items() if count >= threshold]

        # Build template using most common words
        if common_words:
            return f"Related to {' '.join(common_words[:5])}"
        else:
            # Fallback to first description
            return descriptions[0] if descriptions else ""

    def _create_regex_from_pattern(self, pattern_key: str) -> str:
        """
        Create a regex pattern from pattern key
        """
        # Replace "entity" with wildcard
        regex = pattern_key.replace('entity', r'[a-z_]+')
        return f'^{regex}$'

    def _build_description_templates(self, column_catalog: Dict):
        """
        Build reusable description templates based on learned patterns
        """
        logger.info("Building description templates...")

        for pattern in self.learned_patterns:
            self.description_templates[pattern.pattern_name] = pattern.standard_description_template

        logger.info(f"Built {len(self.description_templates)} templates")

    def match_pattern(self, column_name: str) -> Optional[LearnedPattern]:
        """
        Find a learned pattern that matches the column name
        """
        for pattern in sorted(self.learned_patterns, key=lambda p: p.confidence_score, reverse=True):
            if re.match(pattern.pattern_regex, column_name):
                return pattern

        return None

    def get_suggested_description(self, column_name: str, existing_patterns: List[str] = None) -> Optional[str]:
        """
        Get a suggested description based on learned patterns
        """
        # Try to match learned pattern
        learned = self.match_pattern(column_name)
        if learned:
            # Customize template for this specific column
            entity = self._extract_entity_from_column(column_name)
            desc = learned.standard_description_template

            if entity and '{entity}' in desc:
                desc = desc.replace('{entity}', entity)

            return desc

        return None

    def _extract_entity_from_column(self, column_name: str) -> Optional[str]:
        """Extract the entity name from a column"""
        # Remove common suffixes/prefixes
        entity = re.sub(r'_(id|key|at|date|count|amount|price|status|name)$', '', column_name)
        entity = re.sub(r'^(is|has|num|total)_', '', entity)

        return entity.replace('_', ' ') if entity != column_name else None

    def get_business_term_definition(self, term: str) -> Optional[BusinessTerm]:
        """Get definition for a business term"""
        return self.business_glossary.get(term)

    def get_naming_convention_info(self, column_name: str) -> Dict:
        """
        Get information about naming conventions for this column
        """
        info = {
            'follows_conventions': [],
            'suggestions': []
        }

        # Check prefix
        prefix_match = re.match(r'^([a-z]{2,4})_', column_name)
        if prefix_match:
            prefix = prefix_match.group(1)
            if prefix in self.naming_conventions.get('prefixes', {}):
                examples = self.naming_conventions['prefixes'][prefix]
                info['follows_conventions'].append(f"Uses standard prefix '{prefix}_' ({len(examples)} similar columns)")

        # Check suffix
        suffix_match = re.search(r'_([a-z]{2,10})$', column_name)
        if suffix_match:
            suffix = suffix_match.group(1)
            if suffix in self.naming_conventions.get('suffixes', {}):
                examples = self.naming_conventions['suffixes'][suffix]
                info['follows_conventions'].append(f"Uses standard suffix '_{suffix}' ({len(examples)} similar columns)")

        return info

    def export_learned_knowledge(self) -> Dict:
        """Export all learned knowledge as JSON"""
        return {
            'learned_patterns': [
                {
                    'name': p.pattern_name,
                    'regex': p.pattern_regex,
                    'examples': p.example_columns,
                    'template': p.standard_description_template,
                    'confidence': p.confidence_score,
                    'occurrences': p.occurrences
                }
                for p in self.learned_patterns
            ],
            'business_glossary': {
                term: {
                    'definition': bt.definition,
                    'related_columns': bt.related_columns,
                    'confidence': bt.confidence
                }
                for term, bt in self.business_glossary.items()
            },
            'naming_conventions': dict(self.naming_conventions),
            'description_templates': self.description_templates
        }

    def import_learned_knowledge(self, knowledge: Dict):
        """Import previously learned knowledge"""
        # Import patterns
        for p in knowledge.get('learned_patterns', []):
            self.learned_patterns.append(LearnedPattern(
                pattern_regex=p['regex'],
                pattern_name=p['name'],
                example_columns=p['examples'],
                learned_descriptions=[],
                confidence_score=p['confidence'],
                occurrences=p['occurrences'],
                standard_description_template=p['template']
            ))

        # Import business glossary
        for term, info in knowledge.get('business_glossary', {}).items():
            self.business_glossary[term] = BusinessTerm(
                term=term,
                definition=info['definition'],
                related_columns=info['related_columns'],
                confidence=info['confidence']
            )

        # Import naming conventions
        self.naming_conventions = defaultdict(list, knowledge.get('naming_conventions', {}))

        # Import templates
        self.description_templates = knowledge.get('description_templates', {})

        logger.info(f"Imported {len(self.learned_patterns)} patterns, {len(self.business_glossary)} business terms")