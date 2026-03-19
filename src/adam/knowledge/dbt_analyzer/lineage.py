"""
DBT Lineage Tracker
Advanced lineage analysis and column-level tracking for DBT projects
"""

import re
import sqlparse
from typing import Dict, List, Set, Optional, Tuple
from dataclasses import dataclass
import networkx as nx
import json
import logging

logger = logging.getLogger(__name__)

@dataclass
class ColumnLineage:
    """Represents column-level lineage"""
    column_name: str
    source_table: str
    source_column: str
    transformations: List[str]
    dependencies: List[str]


class DBTLineageTracker:
    """
    Track and analyze data lineage at both model and column level
    """

    def __init__(self):
        self.model_graph = nx.DiGraph()
        self.column_lineage: Dict[str, Dict[str, ColumnLineage]] = {}
        self.impact_cache: Dict[str, Set[str]] = {}

    def build_lineage_graph(self, models: Dict, sources: Dict):
        """Build a complete lineage graph from models and sources"""
        # Add sources as nodes
        for source_id, source in sources.items():
            self.model_graph.add_node(
                source_id,
                type="source",
                name=source.name,
                database=source.database,
                schema=source.schema
            )

        # Add models as nodes
        for model_id, model in models.items():
            self.model_graph.add_node(
                model_id,
                type="model",
                name=model.name,
                materialization=model.materialization,
                database=model.database,
                schema=model.schema
            )

            # Add edges for dependencies
            for dep in model.depends_on:
                self.model_graph.add_edge(dep, model_id)

    def extract_column_lineage(self, sql: str, model_name: str) -> Dict[str, ColumnLineage]:
        """
        Extract column-level lineage from SQL
        This is a simplified version - production would need more robust SQL parsing
        """
        column_lineage = {}

        # Parse SQL
        parsed = sqlparse.parse(sql)[0]

        # Extract SELECT columns
        columns = self._extract_select_columns(parsed)

        # Extract FROM/JOIN tables
        tables = self._extract_tables(parsed)

        # Map columns to their sources
        for column in columns:
            # Simple heuristic - in production, use proper SQL AST
            source_info = self._trace_column_source(column, sql, tables)

            column_lineage[column] = ColumnLineage(
                column_name=column,
                source_table=source_info.get("table", "unknown"),
                source_column=source_info.get("column", column),
                transformations=source_info.get("transformations", []),
                dependencies=source_info.get("dependencies", [])
            )

        self.column_lineage[model_name] = column_lineage
        return column_lineage

    def _extract_select_columns(self, parsed_sql) -> List[str]:
        """Extract column names from SELECT statement"""
        columns = []

        # Simple regex pattern for column extraction
        # In production, use proper SQL AST parsing
        select_pattern = r'SELECT\s+(.*?)\s+FROM'
        match = re.search(select_pattern, str(parsed_sql), re.IGNORECASE | re.DOTALL)

        if match:
            select_clause = match.group(1)
            # Split by comma, handle aliases
            for col in select_clause.split(','):
                col = col.strip()
                # Remove alias if present
                if ' AS ' in col.upper():
                    col = col.split(' AS ')[1].strip()
                elif ' ' in col:
                    # Might be implicit alias
                    parts = col.split()
                    if len(parts) == 2:
                        col = parts[1]
                columns.append(col.strip('`"[]'))

        return columns

    def _extract_tables(self, parsed_sql) -> List[str]:
        """Extract table names from FROM and JOIN clauses"""
        tables = []

        # Extract FROM table
        from_pattern = r'FROM\s+([^\s,]+)'
        match = re.search(from_pattern, str(parsed_sql), re.IGNORECASE)
        if match:
            tables.append(match.group(1).strip('`"[]'))

        # Extract JOIN tables
        join_pattern = r'JOIN\s+([^\s]+)'
        for match in re.finditer(join_pattern, str(parsed_sql), re.IGNORECASE):
            tables.append(match.group(1).strip('`"[]'))

        return tables

    def _trace_column_source(self, column: str, sql: str, tables: List[str]) -> Dict:
        """
        Trace the source of a column
        This is simplified - production would need proper SQL parsing
        """
        source_info = {
            "table": tables[0] if tables else "unknown",
            "column": column,
            "transformations": [],
            "dependencies": []
        }

        # Check for transformations
        transformations = [
            ("CASE WHEN", "conditional"),
            ("SUM(", "aggregation"),
            ("COUNT(", "aggregation"),
            ("AVG(", "aggregation"),
            ("MAX(", "aggregation"),
            ("MIN(", "aggregation"),
            ("COALESCE(", "null_handling"),
            ("CAST(", "type_conversion"),
            ("DATE(", "date_transformation"),
            ("CONCAT(", "string_manipulation")
        ]

        sql_upper = sql.upper()
        for pattern, trans_type in transformations:
            if pattern in sql_upper and column.upper() in sql_upper.split(pattern)[1].split(')')[0]:
                source_info["transformations"].append(trans_type)

        # Check for dependencies (simplified)
        if "." in column:
            parts = column.split(".")
            source_info["table"] = parts[0]
            source_info["column"] = parts[1]

        return source_info

    def get_upstream_models(self, model_id: str, levels: Optional[int] = None) -> Set[str]:
        """Get all upstream models up to specified levels"""
        if levels is None:
            # Get all ancestors
            return set(nx.ancestors(self.model_graph, model_id))
        else:
            # Get ancestors up to specified levels
            upstream = set()
            current_level = {model_id}

            for _ in range(levels):
                next_level = set()
                for node in current_level:
                    predecessors = set(self.model_graph.predecessors(node))
                    next_level.update(predecessors)
                    upstream.update(predecessors)
                current_level = next_level

                if not current_level:
                    break

            return upstream

    def get_downstream_models(self, model_id: str, levels: Optional[int] = None) -> Set[str]:
        """Get all downstream models up to specified levels"""
        if levels is None:
            # Get all descendants
            return set(nx.descendants(self.model_graph, model_id))
        else:
            # Get descendants up to specified levels
            downstream = set()
            current_level = {model_id}

            for _ in range(levels):
                next_level = set()
                for node in current_level:
                    successors = set(self.model_graph.successors(node))
                    next_level.update(successors)
                    downstream.update(successors)
                current_level = next_level

                if not current_level:
                    break

            return downstream

    def calculate_impact_analysis(self, model_id: str) -> Dict:
        """
        Calculate the full impact of changing a model
        """
        # Direct impact
        direct_upstream = set(self.model_graph.predecessors(model_id))
        direct_downstream = set(self.model_graph.successors(model_id))

        # Full impact
        all_upstream = self.get_upstream_models(model_id)
        all_downstream = self.get_downstream_models(model_id)

        # Calculate criticality score
        criticality = self._calculate_criticality(model_id, all_downstream)

        return {
            "model_id": model_id,
            "direct_impact": {
                "upstream": list(direct_upstream),
                "downstream": list(direct_downstream),
                "total": len(direct_upstream) + len(direct_downstream)
            },
            "full_impact": {
                "upstream": list(all_upstream),
                "downstream": list(all_downstream),
                "total": len(all_upstream) + len(all_downstream)
            },
            "criticality": criticality,
            "is_leaf": len(direct_downstream) == 0,
            "is_root": len(direct_upstream) == 0,
            "depth": self._get_node_depth(model_id)
        }

    def _calculate_criticality(self, model_id: str, downstream: Set[str]) -> str:
        """Calculate criticality score based on downstream dependencies"""
        downstream_count = len(downstream)

        # Check if any downstream nodes are exposures
        has_exposures = any(
            self.model_graph.nodes[node].get("type") == "exposure"
            for node in downstream
            if node in self.model_graph.nodes
        )

        if has_exposures or downstream_count > 10:
            return "CRITICAL"
        elif downstream_count > 5:
            return "HIGH"
        elif downstream_count > 2:
            return "MEDIUM"
        else:
            return "LOW"

    def _get_node_depth(self, model_id: str) -> int:
        """Calculate the depth of a node in the DAG"""
        if model_id not in self.model_graph:
            return -1

        # Find all paths from root nodes to this node
        roots = [n for n in self.model_graph.nodes() if self.model_graph.in_degree(n) == 0]

        max_depth = 0
        for root in roots:
            try:
                paths = nx.all_simple_paths(self.model_graph, root, model_id)
                for path in paths:
                    max_depth = max(max_depth, len(path) - 1)
            except nx.NetworkXNoPath:
                continue

        return max_depth

    def find_circular_dependencies(self) -> List[List[str]]:
        """Find any circular dependencies in the lineage graph"""
        try:
            cycles = list(nx.simple_cycles(self.model_graph))
            return cycles
        except:
            return []

    def get_lineage_path(self, source_model: str, target_model: str) -> List[List[str]]:
        """Get all paths between two models"""
        try:
            paths = list(nx.all_simple_paths(
                self.model_graph,
                source_model,
                target_model
            ))
            return paths
        except nx.NetworkXNoPath:
            return []

    def generate_lineage_report(self, model_id: str) -> str:
        """Generate a comprehensive lineage report for a model"""
        impact = self.calculate_impact_analysis(model_id)
        node_info = self.model_graph.nodes[model_id]

        report = f"""
# Lineage Report for {node_info.get('name', model_id)}

## Model Information
- **Type**: {node_info.get('type', 'unknown')}
- **Materialization**: {node_info.get('materialization', 'N/A')}
- **Database**: {node_info.get('database', 'N/A')}
- **Schema**: {node_info.get('schema', 'N/A')}
- **Depth in DAG**: {impact['depth']}

## Impact Analysis
### Direct Impact
- **Upstream Dependencies**: {len(impact['direct_impact']['upstream'])}
  - {', '.join(impact['direct_impact']['upstream'][:5])}{'...' if len(impact['direct_impact']['upstream']) > 5 else ''}
- **Downstream Dependencies**: {len(impact['direct_impact']['downstream'])}
  - {', '.join(impact['direct_impact']['downstream'][:5])}{'...' if len(impact['direct_impact']['downstream']) > 5 else ''}

### Full Impact
- **Total Upstream**: {len(impact['full_impact']['upstream'])}
- **Total Downstream**: {len(impact['full_impact']['downstream'])}
- **Criticality Level**: {impact['criticality']}

## Model Characteristics
- **Is Leaf Node**: {impact['is_leaf']}
- **Is Root Node**: {impact['is_root']}
"""

        # Add column lineage if available
        if model_id in self.column_lineage:
            report += "\n## Column Lineage\n"
            for col_name, col_lineage in self.column_lineage[model_id].items():
                report += f"- **{col_name}**\n"
                report += f"  - Source: {col_lineage.source_table}.{col_lineage.source_column}\n"
                if col_lineage.transformations:
                    report += f"  - Transformations: {', '.join(col_lineage.transformations)}\n"

        return report

    def export_lineage_graph(self, output_format: str = "json") -> str:
        """Export the lineage graph in various formats"""
        if output_format == "json":
            # Convert to node-link format for D3.js visualization
            data = nx.node_link_data(self.model_graph)
            return json.dumps(data, indent=2)

        elif output_format == "dot":
            # Export as DOT format for Graphviz
            return nx.nx_pydot.to_pydot(self.model_graph).to_string()

        elif output_format == "cypher":
            # Export as Cypher queries for Neo4j
            cypher_queries = []

            # Create nodes
            for node_id, attrs in self.model_graph.nodes(data=True):
                props = ", ".join([f"{k}: '{v}'" for k, v in attrs.items()])
                cypher_queries.append(
                    f"CREATE (n:{attrs.get('type', 'Model')} {{id: '{node_id}', {props}}})"
                )

            # Create relationships
            for source, target in self.model_graph.edges():
                cypher_queries.append(
                    f"MATCH (a {{id: '{source}'}}), (b {{id: '{target}'}}) "
                    f"CREATE (a)-[:DEPENDS_ON]->(b)"
                )

            return "\n".join(cypher_queries)

        else:
            raise ValueError(f"Unsupported format: {output_format}")