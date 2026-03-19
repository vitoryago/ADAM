"""
Lineage Tracking Service for ADAM
Analyzes and tracks dependencies, data flow, and relationships in projects.

Moved from: src/adam_v2/services/lineage_service.py
"""

import ast
import os
import re
import json
import networkx as nx
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Any
from dataclasses import dataclass, asdict
from enum import Enum
import logging
import yaml
from collections import defaultdict
import sqlparse

logger = logging.getLogger(__name__)

class NodeType(Enum):
    """Types of nodes in the lineage graph"""
    TABLE = "table"
    VIEW = "view"
    MODEL = "model"
    FILE = "file"
    FUNCTION = "function"
    CLASS = "class"
    API = "api"
    SERVICE = "service"
    COMPONENT = "component"
    DATABASE = "database"
    SCHEMA = "schema"
    COLUMN = "column"

class RelationType(Enum):
    """Types of relationships in the lineage graph"""
    DEPENDS_ON = "depends_on"
    CREATES = "creates"
    READS = "reads"
    WRITES = "writes"
    TRANSFORMS = "transforms"
    CALLS = "calls"
    IMPORTS = "imports"
    EXTENDS = "extends"
    IMPLEMENTS = "implements"
    USES = "uses"
    CONTAINS = "contains"

@dataclass
class LineageNode:
    """Represents a node in the lineage graph"""
    id: str
    name: str
    type: NodeType
    path: Optional[str] = None
    schema: Optional[str] = None
    database: Optional[str] = None
    metadata: Dict[str, Any] = None
    description: Optional[str] = None
    tags: List[str] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
        if self.tags is None:
            self.tags = []

@dataclass
class LineageEdge:
    """Represents an edge in the lineage graph"""
    source_id: str
    target_id: str
    relationship: RelationType
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

class LineageAnalyzer:
    """Analyzes code and data to build lineage graphs"""

    def __init__(self):
        self.graph = nx.DiGraph()
        self.nodes: Dict[str, LineageNode] = {}
        self.edges: List[LineageEdge] = []

    def analyze_python_file(self, file_path: Path) -> List[LineageNode]:
        """Analyze a Python file for dependencies and definitions"""
        nodes = []

        try:
            with open(file_path, 'r') as f:
                content = f.read()
                tree = ast.parse(content)

            file_node = LineageNode(
                id=str(file_path),
                name=file_path.name,
                type=NodeType.FILE,
                path=str(file_path)
            )
            nodes.append(file_node)

            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        import_node = LineageNode(
                            id=f"import_{alias.name}",
                            name=alias.name,
                            type=NodeType.FILE
                        )
                        nodes.append(import_node)
                        self.edges.append(LineageEdge(
                            source_id=str(file_path),
                            target_id=f"import_{alias.name}",
                            relationship=RelationType.IMPORTS
                        ))

                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        import_node = LineageNode(
                            id=f"import_{node.module}",
                            name=node.module,
                            type=NodeType.FILE
                        )
                        nodes.append(import_node)
                        self.edges.append(LineageEdge(
                            source_id=str(file_path),
                            target_id=f"import_{node.module}",
                            relationship=RelationType.IMPORTS
                        ))

                elif isinstance(node, ast.ClassDef):
                    class_node = LineageNode(
                        id=f"{file_path}::{node.name}",
                        name=node.name,
                        type=NodeType.CLASS,
                        path=str(file_path),
                        metadata={"lineno": node.lineno}
                    )
                    nodes.append(class_node)
                    self.edges.append(LineageEdge(
                        source_id=str(file_path),
                        target_id=f"{file_path}::{node.name}",
                        relationship=RelationType.CONTAINS
                    ))

                elif isinstance(node, ast.FunctionDef):
                    func_node = LineageNode(
                        id=f"{file_path}::{node.name}",
                        name=node.name,
                        type=NodeType.FUNCTION,
                        path=str(file_path),
                        metadata={"lineno": node.lineno}
                    )
                    nodes.append(func_node)
                    self.edges.append(LineageEdge(
                        source_id=str(file_path),
                        target_id=f"{file_path}::{node.name}",
                        relationship=RelationType.CONTAINS
                    ))

        except Exception as e:
            logger.error(f"Error analyzing Python file {file_path}: {e}")

        return nodes

    def analyze_sql_file(self, file_path: Path) -> List[LineageNode]:
        """Analyze SQL file for table dependencies"""
        nodes = []

        try:
            with open(file_path, 'r') as f:
                content = f.read()

            statements = sqlparse.split(content)

            for statement in statements:
                parsed = sqlparse.parse(statement)[0] if sqlparse.parse(statement) else None
                if not parsed:
                    continue

                tables = self._extract_tables_from_sql(str(parsed))
                for table in tables:
                    table_node = LineageNode(
                        id=f"table_{table}",
                        name=table,
                        type=NodeType.TABLE
                    )
                    nodes.append(table_node)

        except Exception as e:
            logger.error(f"Error analyzing SQL file {file_path}: {e}")

        return nodes

    def analyze_dbt_model(self, file_path: Path) -> List[LineageNode]:
        """Analyze DBT model for dependencies"""
        nodes = []

        try:
            with open(file_path, 'r') as f:
                content = f.read()

            model_name = file_path.stem
            model_node = LineageNode(
                id=f"dbt_model_{model_name}",
                name=model_name,
                type=NodeType.MODEL,
                path=str(file_path)
            )
            nodes.append(model_node)

            ref_pattern = r"ref\(['\"]([\w_]+)['\"]\)"
            refs = re.findall(ref_pattern, content)
            for ref in refs:
                ref_node = LineageNode(
                    id=f"dbt_model_{ref}",
                    name=ref,
                    type=NodeType.MODEL
                )
                nodes.append(ref_node)
                self.edges.append(LineageEdge(
                    source_id=f"dbt_model_{model_name}",
                    target_id=f"dbt_model_{ref}",
                    relationship=RelationType.DEPENDS_ON
                ))

            source_pattern = r"source\(['\"]([\w_]+)['\"],\s*['\"]([\w_]+)['\"]\)"
            sources = re.findall(source_pattern, content)
            for schema, table in sources:
                source_node = LineageNode(
                    id=f"source_{schema}_{table}",
                    name=table,
                    type=NodeType.TABLE,
                    schema=schema
                )
                nodes.append(source_node)
                self.edges.append(LineageEdge(
                    source_id=f"dbt_model_{model_name}",
                    target_id=f"source_{schema}_{table}",
                    relationship=RelationType.READS
                ))

        except Exception as e:
            logger.error(f"Error analyzing DBT model {file_path}: {e}")

        return nodes

    def _extract_tables_from_sql(self, sql: str) -> List[str]:
        """Extract table names from SQL statement"""
        tables = []

        patterns = [
            r'FROM\s+([^\s,]+)',
            r'JOIN\s+([^\s,]+)',
            r'INTO\s+([^\s,]+)',
            r'UPDATE\s+([^\s,]+)',
            r'TABLE\s+([^\s,]+)'
        ]

        for pattern in patterns:
            matches = re.findall(pattern, sql, re.IGNORECASE)
            tables.extend(matches)

        return list(set(tables))

class LineageService:
    """Main service for lineage tracking and visualization"""

    def __init__(self):
        self.analyzer = LineageAnalyzer()
        self.graph = nx.DiGraph()

    def analyze_directory(self, directory: Path, pattern: str = "*") -> Dict[str, Any]:
        """Analyze all files in a directory matching pattern"""
        logger.info(f"Analyzing directory: {directory} with pattern: {pattern}")

        nodes = []
        file_count = 0

        for file_path in directory.rglob(pattern):
            if file_path.is_file():
                file_count += 1

                if file_path.suffix == '.py':
                    nodes.extend(self.analyzer.analyze_python_file(file_path))
                elif file_path.suffix == '.sql':
                    nodes.extend(self.analyzer.analyze_sql_file(file_path))
                    nodes.extend(self.analyzer.analyze_dbt_model(file_path))

        for node in nodes:
            if node.id not in self.analyzer.nodes:
                self.analyzer.nodes[node.id] = node
                self.graph.add_node(node.id, **asdict(node))

        for edge in self.analyzer.edges:
            self.graph.add_edge(edge.source_id, edge.target_id,
                              relationship=edge.relationship.value,
                              **edge.metadata)

        return {
            "files_analyzed": file_count,
            "nodes": len(self.graph.nodes),
            "edges": len(self.graph.edges),
            "components": list(nx.weakly_connected_components(self.graph))
        }

    def get_lineage(self, node_id: str, direction: str = "both", depth: int = 3) -> Dict[str, Any]:
        """Get lineage for a specific node"""
        if node_id not in self.graph:
            return {"error": f"Node {node_id} not found"}

        upstream = set()
        downstream = set()

        if direction in ["upstream", "both"]:
            for node in nx.ancestors(self.graph, node_id):
                if nx.shortest_path_length(self.graph, node, node_id) <= depth:
                    upstream.add(node)

        if direction in ["downstream", "both"]:
            for node in nx.descendants(self.graph, node_id):
                if nx.shortest_path_length(self.graph, node_id, node) <= depth:
                    downstream.add(node)

        relevant_nodes = {node_id} | upstream | downstream
        subgraph = self.graph.subgraph(relevant_nodes)

        return {
            "node": self.analyzer.nodes.get(node_id),
            "upstream": [self.analyzer.nodes.get(n) for n in upstream],
            "downstream": [self.analyzer.nodes.get(n) for n in downstream],
            "graph": {
                "nodes": [{"id": n, **self.graph.nodes[n]} for n in subgraph.nodes],
                "edges": [{"source": s, "target": t, **self.graph.edges[s, t]}
                         for s, t in subgraph.edges]
            }
        }

    def get_impact_analysis(self, node_id: str) -> Dict[str, Any]:
        """Analyze the impact of changes to a node"""
        if node_id not in self.graph:
            return {"error": f"Node {node_id} not found"}

        downstream = nx.descendants(self.graph, node_id)

        impact_levels = defaultdict(list)
        for node in downstream:
            try:
                distance = nx.shortest_path_length(self.graph, node_id, node)
                impact_levels[distance].append(node)
            except nx.NetworkXNoPath:
                continue

        return {
            "node": self.analyzer.nodes.get(node_id),
            "total_impacted": len(downstream),
            "impact_levels": dict(impact_levels),
            "critical_paths": self._find_critical_paths(node_id)
        }

    def _find_critical_paths(self, node_id: str, max_paths: int = 5) -> List[List[str]]:
        """Find critical dependency paths from a node"""
        paths = []

        leaf_nodes = [n for n in self.graph.nodes if self.graph.out_degree(n) == 0]

        for leaf in leaf_nodes:
            try:
                for path in nx.all_simple_paths(self.graph, node_id, leaf):
                    paths.append(path)
                    if len(paths) >= max_paths:
                        return paths
            except nx.NetworkXNoPath:
                continue

        return paths

    def export_lineage(self, format: str = "json") -> str:
        """Export lineage graph in various formats"""
        if format == "json":
            data = {
                "nodes": [asdict(node) for node in self.analyzer.nodes.values()],
                "edges": [asdict(edge) for edge in self.analyzer.edges]
            }
            return json.dumps(data, indent=2, default=str)

        elif format == "dot":
            return nx.drawing.nx_pydot.to_pydot(self.graph).to_string()

        elif format == "mermaid":
            lines = ["graph TD"]
            for source, target in self.graph.edges:
                source_label = self.analyzer.nodes[source].name if source in self.analyzer.nodes else source
                target_label = self.analyzer.nodes[target].name if target in self.analyzer.nodes else target
                lines.append(f"    {source_label} --> {target_label}")
            return "\n".join(lines)

        else:
            return json.dumps({"error": f"Unknown format: {format}"})

    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about the lineage graph"""
        return {
            "total_nodes": len(self.graph.nodes),
            "total_edges": len(self.graph.edges),
            "node_types": self._count_node_types(),
            "relationship_types": self._count_relationship_types(),
            "connected_components": nx.number_weakly_connected_components(self.graph),
            "average_degree": sum(dict(self.graph.degree()).values()) / len(self.graph.nodes) if self.graph.nodes else 0,
            "most_connected": self._get_most_connected_nodes(),
            "orphan_nodes": self._find_orphan_nodes()
        }

    def _count_node_types(self) -> Dict[str, int]:
        counts = defaultdict(int)
        for node in self.analyzer.nodes.values():
            counts[node.type.value] += 1
        return dict(counts)

    def _count_relationship_types(self) -> Dict[str, int]:
        counts = defaultdict(int)
        for edge in self.analyzer.edges:
            counts[edge.relationship.value] += 1
        return dict(counts)

    def _get_most_connected_nodes(self, limit: int = 10) -> List[Tuple[str, int]]:
        degrees = dict(self.graph.degree())
        sorted_nodes = sorted(degrees.items(), key=lambda x: x[1], reverse=True)
        return sorted_nodes[:limit]

    def _find_orphan_nodes(self) -> List[str]:
        return [n for n in self.graph.nodes if self.graph.degree(n) == 0]
