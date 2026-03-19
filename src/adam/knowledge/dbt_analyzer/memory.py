"""
DBT Memory System
Specialized memory storage and retrieval for DBT projects
"""

import json
import hashlib
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import chromadb
from chromadb.config import Settings
import numpy as np
from pathlib import Path
import logging

from .parser import DBTModel, DBTSource, DBTTest, DBTExposure, DBTManifestParser

logger = logging.getLogger(__name__)

@dataclass
class DBTMemoryEntry:
    """Represents a memory entry for DBT artifacts"""
    id: str
    type: str  # model, source, test, exposure, pattern, optimization
    content: str
    metadata: Dict[str, Any]
    embeddings: Optional[List[float]] = None
    timestamp: str = ""
    project_id: Optional[str] = None
    relevance_score: float = 1.0

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()


class DBTMemorySystem:
    """
    Advanced memory system specifically designed for DBT projects
    Stores models, patterns, optimizations, and lineage information
    """

    def __init__(self, memory_path: str = "./data/dbt_memory", project_id: Optional[str] = None):
        self.memory_path = Path(memory_path)
        self.memory_path.mkdir(parents=True, exist_ok=True)
        self.project_id = project_id or "default"

        # Initialize ChromaDB client with persistent storage
        self.chroma_client = chromadb.PersistentClient(
            path=str(self.memory_path / "chroma"),
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )

        # Create specialized collections for different DBT artifacts
        self.collections = {
            "models": self._get_or_create_collection("dbt_models"),
            "sql_patterns": self._get_or_create_collection("sql_patterns"),
            "optimizations": self._get_or_create_collection("optimizations"),
            "lineage": self._get_or_create_collection("lineage"),
            "documentation": self._get_or_create_collection("documentation"),
            "tests": self._get_or_create_collection("tests")
        }

        # Cache for frequently accessed data
        self.cache = {
            "model_graph": {},
            "pattern_library": {},
            "optimization_rules": {}
        }

        # Load optimization rules
        self._load_optimization_rules()

    def _get_or_create_collection(self, name: str):
        """Get or create a ChromaDB collection"""
        collection_name = f"{name}_{self.project_id}"
        try:
            return self.chroma_client.get_collection(collection_name)
        except:
            return self.chroma_client.create_collection(
                name=collection_name,
                metadata={"project_id": self.project_id}
            )

    def ingest_dbt_project(self, parser: DBTManifestParser) -> Dict[str, int]:
        """
        Ingest an entire DBT project into memory
        Returns statistics about what was ingested
        """
        stats = {
            "models": 0,
            "sources": 0,
            "tests": 0,
            "exposures": 0,
            "patterns": 0,
            "lineage_links": 0
        }

        # Ingest models
        for model_id, model in parser.models.items():
            self._store_model(model)
            stats["models"] += 1

            # Extract and store SQL patterns
            patterns = self._extract_sql_patterns(model.compiled_sql)
            for pattern in patterns:
                self._store_pattern(pattern, model_id)
                stats["patterns"] += 1

            # Store lineage information
            lineage = parser.get_model_lineage(model.name)
            self._store_lineage(model_id, lineage)
            stats["lineage_links"] += len(lineage["upstream"]) + len(lineage["downstream"])

        # Ingest sources
        for source_id, source in parser.sources.items():
            self._store_source(source)
            stats["sources"] += 1

        # Ingest tests
        for test_id, test in parser.tests.items():
            self._store_test(test)
            stats["tests"] += 1

        # Ingest exposures
        for exposure_id, exposure in parser.exposures.items():
            self._store_exposure(exposure)
            stats["exposures"] += 1

        # Build and cache the model graph
        self._build_model_graph(parser)

        logger.info(f"Ingested DBT project: {stats}")
        return stats

    def _store_model(self, model: DBTModel):
        """Store a DBT model in memory"""
        model_doc = f"""
Model: {model.name}
Description: {model.description}
Materialization: {model.materialization}
Database: {model.database}
Schema: {model.schema}
Tags: {', '.join(model.tags)}

SQL:
{model.compiled_sql}

Columns:
{json.dumps(model.columns, indent=2)}

Config:
{json.dumps(model.config, indent=2)}
"""

        metadata = {
            "type": "model",
            "model_id": model.unique_id,
            "name": model.name,
            "materialization": model.materialization,
            "database": model.database or "",
            "schema": model.schema or "",
            "tags": json.dumps(model.tags),
            "is_incremental": model.is_incremental,
            "depends_on": json.dumps(model.depends_on),
            "timestamp": datetime.now().isoformat()
        }

        self.collections["models"].upsert(
            ids=[model.unique_id],
            documents=[model_doc],
            metadatas=[metadata]
        )

    def _store_source(self, source: DBTSource):
        """Store a DBT source in memory"""
        source_doc = f"""
Source: {source.source_name}.{source.name}
Description: {source.description}
Database: {source.database}
Schema: {source.schema}
Tables: {json.dumps(source.tables, indent=2)}
Freshness: {json.dumps(source.freshness, indent=2) if source.freshness else 'Not configured'}
"""

        metadata = {
            "type": "source",
            "source_id": source.unique_id,
            "name": source.name,
            "source_name": source.source_name,
            "database": source.database or "",
            "schema": source.schema or "",
            "timestamp": datetime.now().isoformat()
        }

        self.collections["models"].upsert(
            ids=[source.unique_id],
            documents=[source_doc],
            metadatas=[metadata]
        )

    def _store_test(self, test: DBTTest):
        """Store a DBT test in memory"""
        test_doc = f"""
Test: {test.name}
Type: {test.test_type}
Severity: {test.severity}
Column: {test.column_name or 'N/A'}
Depends on: {', '.join(test.depends_on)}

SQL:
{test.compiled_sql}
"""

        metadata = {
            "type": "test",
            "test_id": test.unique_id,
            "name": test.name,
            "test_type": test.test_type,
            "severity": test.severity,
            "column_name": test.column_name or "",
            "depends_on": json.dumps(test.depends_on),
            "timestamp": datetime.now().isoformat()
        }

        self.collections["tests"].upsert(
            ids=[test.unique_id],
            documents=[test_doc],
            metadatas=[metadata]
        )

    def _store_exposure(self, exposure: DBTExposure):
        """Store a DBT exposure in memory"""
        exposure_doc = f"""
Exposure: {exposure.name}
Type: {exposure.type}
Description: {exposure.description}
Owner: {json.dumps(exposure.owner)}
URL: {exposure.url or 'N/A'}
Depends on: {', '.join(exposure.depends_on)}
"""

        metadata = {
            "type": "exposure",
            "exposure_id": exposure.unique_id,
            "name": exposure.name,
            "exposure_type": exposure.type,
            "depends_on": json.dumps(exposure.depends_on),
            "timestamp": datetime.now().isoformat()
        }

        self.collections["models"].upsert(
            ids=[exposure.unique_id],
            documents=[exposure_doc],
            metadatas=[metadata]
        )

    def _extract_sql_patterns(self, sql: str) -> List[Dict[str, Any]]:
        """
        Extract common SQL patterns from compiled SQL
        These patterns help identify optimization opportunities
        """
        patterns = []

        # Window functions
        if "OVER (" in sql.upper():
            patterns.append({
                "type": "window_function",
                "description": "Uses window functions for analytics",
                "optimization_hint": "Consider materialization for complex window operations"
            })

        # CTEs
        if "WITH " in sql.upper():
            cte_count = sql.upper().count("WITH ")
            patterns.append({
                "type": "cte",
                "count": cte_count,
                "description": f"Uses {cte_count} CTE(s)",
                "optimization_hint": "Complex CTEs might benefit from staging models"
            })

        # Joins
        join_types = ["INNER JOIN", "LEFT JOIN", "RIGHT JOIN", "FULL JOIN", "CROSS JOIN"]
        for join_type in join_types:
            if join_type in sql.upper():
                count = sql.upper().count(join_type)
                patterns.append({
                    "type": "join",
                    "join_type": join_type,
                    "count": count,
                    "description": f"Contains {count} {join_type}(s)",
                    "optimization_hint": "Multiple joins might benefit from incremental materialization"
                })

        # Aggregations
        agg_functions = ["SUM(", "COUNT(", "AVG(", "MAX(", "MIN(", "GROUP BY"]
        for agg in agg_functions:
            if agg in sql.upper():
                patterns.append({
                    "type": "aggregation",
                    "function": agg.replace("(", ""),
                    "description": f"Uses {agg.replace('(', '')} aggregation",
                    "optimization_hint": "Heavy aggregations benefit from table materialization"
                })

        # Subqueries
        if "SELECT" in sql.upper() and sql.upper().count("SELECT") > 1:
            patterns.append({
                "type": "subquery",
                "count": sql.upper().count("SELECT") - 1,
                "description": "Contains subqueries",
                "optimization_hint": "Consider refactoring subqueries into CTEs or staging models"
            })

        return patterns

    def _store_pattern(self, pattern: Dict[str, Any], model_id: str):
        """Store a SQL pattern in memory"""
        pattern_id = hashlib.md5(f"{model_id}_{pattern['type']}_{pattern.get('function', '')}".encode()).hexdigest()

        pattern_doc = f"""
Pattern Type: {pattern['type']}
Description: {pattern['description']}
Model: {model_id}
Optimization Hint: {pattern.get('optimization_hint', 'N/A')}
Details: {json.dumps(pattern)}
"""

        metadata = {
            "type": "sql_pattern",
            "pattern_type": pattern['type'],
            "model_id": model_id,
            "timestamp": datetime.now().isoformat()
        }

        self.collections["sql_patterns"].upsert(
            ids=[pattern_id],
            documents=[pattern_doc],
            metadatas=[metadata]
        )

    def _store_lineage(self, model_id: str, lineage: Dict[str, List[str]]):
        """Store lineage information for a model"""
        lineage_doc = f"""
Model: {model_id}
Upstream Dependencies: {', '.join(lineage['upstream'])}
Downstream Dependencies: {', '.join(lineage['downstream'])}
Total Dependencies: {len(lineage['upstream']) + len(lineage['downstream'])}
"""

        metadata = {
            "type": "lineage",
            "model_id": model_id,
            "upstream_count": len(lineage['upstream']),
            "downstream_count": len(lineage['downstream']),
            "upstream": json.dumps(lineage['upstream']),
            "downstream": json.dumps(lineage['downstream']),
            "timestamp": datetime.now().isoformat()
        }

        self.collections["lineage"].upsert(
            ids=[f"lineage_{model_id}"],
            documents=[lineage_doc],
            metadatas=[metadata]
        )

    def _build_model_graph(self, parser: DBTManifestParser):
        """Build and cache the complete model dependency graph"""
        graph = {}

        for model_id, model in parser.models.items():
            lineage = parser.get_model_lineage(model.name)
            graph[model_id] = {
                "name": model.name,
                "type": "model",
                "materialization": model.materialization,
                "upstream": lineage["upstream"],
                "downstream": lineage["downstream"],
                "depth": 0  # Will be calculated
            }

        # Calculate depth for each node (distance from sources)
        self._calculate_graph_depth(graph)

        self.cache["model_graph"] = graph

    def _calculate_graph_depth(self, graph: Dict):
        """Calculate the depth of each node in the DAG"""
        # Find root nodes (no upstream dependencies)
        roots = [node_id for node_id, node in graph.items() if not node["upstream"]]

        # BFS to calculate depth
        from collections import deque
        queue = deque([(root, 0) for root in roots])
        visited = set()

        while queue:
            node_id, depth = queue.popleft()
            if node_id in visited:
                continue

            visited.add(node_id)
            if node_id in graph:
                graph[node_id]["depth"] = depth

                # Add downstream nodes to queue
                for downstream in graph[node_id]["downstream"]:
                    if downstream not in visited:
                        queue.append((downstream, depth + 1))

    def _load_optimization_rules(self):
        """Load Redshift and Snowflake optimization rules"""
        self.cache["optimization_rules"] = {
            "redshift": {
                "distribution": {
                    "pattern": "JOIN",
                    "suggestion": "Use DISTKEY on join columns for large tables",
                    "example": "ALTER TABLE {table} ALTER DISTKEY {column};"
                },
                "sort_keys": {
                    "pattern": "WHERE.*DATE|TIMESTAMP",
                    "suggestion": "Use SORTKEY on date/timestamp columns for time-series data",
                    "example": "ALTER TABLE {table} ALTER SORTKEY ({column});"
                },
                "compression": {
                    "pattern": "CREATE TABLE",
                    "suggestion": "Use ENCODE AUTO for automatic compression",
                    "example": "CREATE TABLE {table} (...) ENCODE AUTO;"
                },
                "materialized_views": {
                    "pattern": "AGGREGATION",
                    "suggestion": "Consider materialized views for heavy aggregations",
                    "example": "CREATE MATERIALIZED VIEW {view_name} AS SELECT ..."
                }
            },
            "snowflake": {
                "clustering": {
                    "pattern": "WHERE.*=|IN",
                    "suggestion": "Use clustering keys on frequently filtered columns",
                    "example": "ALTER TABLE {table} CLUSTER BY ({columns});"
                },
                "partitioning": {
                    "pattern": "DATE|TIMESTAMP",
                    "suggestion": "Consider partitioning for large time-series tables",
                    "example": "CREATE TABLE {table} ... PARTITION BY DATE({column});"
                },
                "warehouse_sizing": {
                    "pattern": "COMPLEX_QUERY",
                    "suggestion": "Use appropriate warehouse size for query complexity",
                    "example": "USE WAREHOUSE {size}_WAREHOUSE;"
                },
                "result_caching": {
                    "pattern": "REPEATED_QUERY",
                    "suggestion": "Leverage result caching for repeated queries",
                    "example": "ALTER SESSION SET USE_CACHED_RESULT = TRUE;"
                }
            }
        }

    def search_models(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Search for models based on a query
        Uses semantic search with ChromaDB
        """
        results = self.collections["models"].query(
            query_texts=[query],
            n_results=limit,
            where={"type": "model"}
        )

        return self._format_search_results(results)

    def search_patterns(self, pattern_type: Optional[str] = None, limit: int = 10) -> List[Dict[str, Any]]:
        """Search for SQL patterns"""
        where_clause = {"type": "sql_pattern"}
        if pattern_type:
            where_clause["pattern_type"] = pattern_type

        results = self.collections["sql_patterns"].query(
            query_texts=[""],  # Get all if no specific query
            n_results=limit,
            where=where_clause
        )

        return self._format_search_results(results)

    def get_optimization_suggestions(self, model_name: str, platform: str = "redshift") -> List[Dict[str, str]]:
        """
        Get optimization suggestions for a specific model
        """
        # Search for the model
        model_results = self.search_models(model_name, limit=1)
        if not model_results:
            return []

        model_id = model_results[0]["metadata"]["model_id"]

        # Get patterns for this model
        patterns = self.collections["sql_patterns"].query(
            query_texts=[""],
            n_results=20,
            where={"model_id": model_id}
        )

        suggestions = []
        rules = self.cache["optimization_rules"].get(platform, {})

        for i, pattern_metadata in enumerate(patterns["metadatas"][0] if patterns["metadatas"] else []):
            pattern_type = pattern_metadata.get("pattern_type", "")

            # Match patterns with optimization rules
            for rule_name, rule in rules.items():
                if pattern_type.upper() in rule["pattern"].upper():
                    suggestions.append({
                        "model": model_name,
                        "pattern": pattern_type,
                        "rule": rule_name,
                        "suggestion": rule["suggestion"],
                        "example": rule["example"]
                    })

        return suggestions

    def get_lineage_impact(self, model_name: str) -> Dict[str, Any]:
        """
        Get the potential impact of changing a model
        """
        # Search for the model
        model_results = self.search_models(model_name, limit=1)
        if not model_results:
            return {"error": "Model not found"}

        model_id = model_results[0]["metadata"]["model_id"]

        # Get lineage information
        lineage_results = self.collections["lineage"].query(
            query_texts=[""],
            n_results=1,
            where={"model_id": model_id}
        )

        if not lineage_results["metadatas"]:
            return {"error": "No lineage information found"}

        lineage_metadata = lineage_results["metadatas"][0][0]

        upstream = json.loads(lineage_metadata.get("upstream", "[]"))
        downstream = json.loads(lineage_metadata.get("downstream", "[]"))

        return {
            "model": model_name,
            "direct_upstream_impact": len(upstream),
            "direct_downstream_impact": len(downstream),
            "upstream_models": upstream,
            "downstream_models": downstream,
            "total_impact": len(upstream) + len(downstream),
            "risk_level": "HIGH" if len(downstream) > 5 else "MEDIUM" if len(downstream) > 2 else "LOW"
        }

    def _format_search_results(self, results) -> List[Dict[str, Any]]:
        """Format ChromaDB search results"""
        formatted = []

        if results and results["ids"] and results["ids"][0]:
            for i, id in enumerate(results["ids"][0]):
                formatted.append({
                    "id": id,
                    "document": results["documents"][0][i] if results["documents"] else "",
                    "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                    "distance": results["distances"][0][i] if results["distances"] else 0
                })

        return formatted

    def export_knowledge_base(self, output_path: str):
        """Export the entire DBT knowledge base to a file"""
        knowledge = {
            "project_id": self.project_id,
            "timestamp": datetime.now().isoformat(),
            "model_graph": self.cache.get("model_graph", {}),
            "optimization_rules": self.cache.get("optimization_rules", {}),
            "statistics": {
                "total_models": len(self.collections["models"].get()),
                "total_patterns": len(self.collections["sql_patterns"].get()),
                "total_tests": len(self.collections["tests"].get())
            }
        }

        with open(output_path, 'w') as f:
            json.dump(knowledge, f, indent=2)

        logger.info(f"Exported knowledge base to {output_path}")