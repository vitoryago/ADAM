#!/usr/bin/env python
"""
DBT Analyzer CLI
Command-line interface for analyzing DBT projects with ADAM
"""

import argparse
import sys
import json
from pathlib import Path
import logging
from typing import Optional

from .parser import DBTManifestParser
from .memory import DBTMemorySystem
from .lineage import DBTLineageTracker
from .optimizer import SQLOptimizer
from .documentation import DBTDocGenerator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def analyze_project(args):
    """Analyze a DBT project and store in memory"""
    print(f"\n🔍 Analyzing DBT project at: {args.project_path}\n")

    # Initialize parser
    parser = DBTManifestParser(args.project_path)

    # Parse manifest
    if not parser.parse():
        print("❌ Failed to parse manifest.json")
        print("💡 Tip: Run 'dbt compile' or 'dbt run' first to generate manifest.json")
        return 1

    # Initialize memory system
    memory = DBTMemorySystem(
        memory_path=args.memory_path,
        project_id=args.project_id
    )

    # Ingest project
    stats = memory.ingest_dbt_project(parser)

    print("✅ Successfully analyzed DBT project!")
    print("\n📊 Statistics:")
    for key, value in stats.items():
        print(f"  - {key}: {value}")

    # Export if requested
    if args.export:
        output_file = args.export
        memory.export_knowledge_base(output_file)
        print(f"\n📁 Knowledge base exported to: {output_file}")

    return 0


def document_model(args):
    """Generate documentation for a specific model"""
    print(f"\n📝 Generating documentation for model: {args.model_name}\n")

    # Initialize components
    parser = DBTManifestParser(args.project_path)
    if not parser.parse():
        print("❌ Failed to parse manifest.json")
        return 1

    lineage_tracker = DBTLineageTracker()
    lineage_tracker.build_lineage_graph(parser.models, parser.sources)

    optimizer = SQLOptimizer()

    doc_generator = DBTDocGenerator(parser, lineage_tracker, optimizer)

    # Generate documentation
    doc = doc_generator.generate_model_documentation(
        args.model_name,
        include_lineage=not args.no_lineage,
        include_tests=not args.no_tests,
        include_optimizations=not args.no_optimizations,
        platform=args.platform
    )

    # Output documentation
    if args.output:
        with open(args.output, 'w') as f:
            f.write(doc)
        print(f"✅ Documentation saved to: {args.output}")
    else:
        print(doc)

    # Export as YAML if requested
    if args.yaml:
        yaml_content = doc_generator.export_to_yaml(args.model_name)
        yaml_file = args.yaml
        with open(yaml_file, 'w') as f:
            f.write(yaml_content)
        print(f"✅ YAML schema saved to: {yaml_file}")

    return 0


def analyze_lineage(args):
    """Analyze lineage for a model"""
    print(f"\n🔗 Analyzing lineage for: {args.model_name}\n")

    # Initialize components
    parser = DBTManifestParser(args.project_path)
    if not parser.parse():
        print("❌ Failed to parse manifest.json")
        return 1

    lineage_tracker = DBTLineageTracker()
    lineage_tracker.build_lineage_graph(parser.models, parser.sources)

    # Find model
    model_id = None
    for mid, model in parser.models.items():
        if model.name == args.model_name:
            model_id = mid
            break

    if not model_id:
        print(f"❌ Model '{args.model_name}' not found")
        return 1

    # Calculate impact
    impact = lineage_tracker.calculate_impact_analysis(model_id)

    print(f"📊 Lineage Analysis for {args.model_name}")
    print(f"{'=' * 50}")

    print(f"\n🔼 Upstream Dependencies: {len(impact['direct_impact']['upstream'])}")
    for dep in impact['direct_impact']['upstream'][:10]:
        model_name = parser.models.get(dep, {}).name if dep in parser.models else dep
        print(f"  └─ {model_name}")

    print(f"\n🔽 Downstream Dependencies: {len(impact['direct_impact']['downstream'])}")
    for dep in impact['direct_impact']['downstream'][:10]:
        model_name = parser.models.get(dep, {}).name if dep in parser.models else dep
        print(f"  └─ {model_name}")

    print(f"\n📈 Impact Summary:")
    print(f"  - Total Upstream: {len(impact['full_impact']['upstream'])}")
    print(f"  - Total Downstream: {len(impact['full_impact']['downstream'])}")
    print(f"  - Criticality: {impact['criticality']}")
    print(f"  - DAG Depth: {impact['depth']}")

    # Export if requested
    if args.export:
        export_format = args.format or "json"
        graph_data = lineage_tracker.export_lineage_graph(export_format)

        with open(args.export, 'w') as f:
            f.write(graph_data)
        print(f"\n✅ Lineage graph exported to: {args.export} (format: {export_format})")

    return 0


def optimize_sql(args):
    """Get optimization suggestions for a model"""
    print(f"\n⚡ Analyzing optimizations for: {args.model_name}\n")

    # Initialize components
    parser = DBTManifestParser(args.project_path)
    if not parser.parse():
        print("❌ Failed to parse manifest.json")
        return 1

    optimizer = SQLOptimizer()

    # Get model SQL
    sql = parser.get_model_sql(args.model_name, compiled=True)
    if not sql:
        print(f"❌ Model '{args.model_name}' not found")
        return 1

    # Generate optimization report
    report = optimizer.generate_optimization_report(
        sql,
        args.model_name,
        platform=args.platform
    )

    # Output report
    if args.output:
        with open(args.output, 'w') as f:
            f.write(report)
        print(f"✅ Optimization report saved to: {args.output}")
    else:
        print(report)

    return 0


def search_models(args):
    """Search for models in memory"""
    print(f"\n🔎 Searching for: {args.query}\n")

    # Initialize memory system
    memory = DBTMemorySystem(
        memory_path=args.memory_path,
        project_id=args.project_id
    )

    # Search models
    results = memory.search_models(args.query, limit=args.limit)

    if not results:
        print("No models found matching your query")
        return 0

    print(f"Found {len(results)} models:\n")
    for i, result in enumerate(results, 1):
        metadata = result["metadata"]
        print(f"{i}. {metadata.get('name', 'Unknown')}")
        print(f"   Type: {metadata.get('materialization', 'unknown')}")
        print(f"   Database: {metadata.get('database', 'N/A')}.{metadata.get('schema', 'N/A')}")
        print(f"   Distance: {result.get('distance', 0):.3f}")
        print()

    return 0


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="DBT Analyzer - Intelligent DBT project analysis with ADAM",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze a DBT project
  python -m adam_v2.dbt_analyzer.cli analyze /path/to/dbt/project

  # Generate documentation for a model
  python -m adam_v2.dbt_analyzer.cli document my_model --project-path /path/to/dbt

  # Analyze lineage
  python -m adam_v2.dbt_analyzer.cli lineage my_model --project-path /path/to/dbt

  # Get optimization suggestions
  python -m adam_v2.dbt_analyzer.cli optimize my_model --platform snowflake

  # Search models in memory
  python -m adam_v2.dbt_analyzer.cli search "customer orders"
        """
    )

    # Global arguments
    parser.add_argument(
        "--project-path",
        default=".",
        help="Path to DBT project (default: current directory)"
    )
    parser.add_argument(
        "--memory-path",
        default="./data/dbt_memory",
        help="Path to memory storage (default: ./data/dbt_memory)"
    )
    parser.add_argument(
        "--project-id",
        default="default",
        help="Project ID for memory isolation (default: default)"
    )

    # Subcommands
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Analyze command
    analyze_parser = subparsers.add_parser(
        "analyze",
        help="Analyze a DBT project and store in memory"
    )
    analyze_parser.add_argument(
        "project_path",
        help="Path to DBT project"
    )
    analyze_parser.add_argument(
        "--export",
        help="Export knowledge base to JSON file"
    )

    # Document command
    doc_parser = subparsers.add_parser(
        "document",
        help="Generate documentation for a model"
    )
    doc_parser.add_argument(
        "model_name",
        help="Name of the model to document"
    )
    doc_parser.add_argument(
        "--output", "-o",
        help="Output file for documentation"
    )
    doc_parser.add_argument(
        "--yaml",
        help="Export as schema.yml format"
    )
    doc_parser.add_argument(
        "--platform",
        choices=["redshift", "snowflake"],
        default="redshift",
        help="Target platform for optimizations"
    )
    doc_parser.add_argument(
        "--no-lineage",
        action="store_true",
        help="Skip lineage information"
    )
    doc_parser.add_argument(
        "--no-tests",
        action="store_true",
        help="Skip test information"
    )
    doc_parser.add_argument(
        "--no-optimizations",
        action="store_true",
        help="Skip optimization suggestions"
    )

    # Lineage command
    lineage_parser = subparsers.add_parser(
        "lineage",
        help="Analyze lineage for a model"
    )
    lineage_parser.add_argument(
        "model_name",
        help="Name of the model"
    )
    lineage_parser.add_argument(
        "--export",
        help="Export lineage graph to file"
    )
    lineage_parser.add_argument(
        "--format",
        choices=["json", "dot", "cypher"],
        default="json",
        help="Export format"
    )

    # Optimize command
    optimize_parser = subparsers.add_parser(
        "optimize",
        help="Get optimization suggestions for a model"
    )
    optimize_parser.add_argument(
        "model_name",
        help="Name of the model"
    )
    optimize_parser.add_argument(
        "--platform",
        choices=["redshift", "snowflake"],
        default="redshift",
        help="Target platform"
    )
    optimize_parser.add_argument(
        "--output", "-o",
        help="Output file for report"
    )

    # Search command
    search_parser = subparsers.add_parser(
        "search",
        help="Search for models in memory"
    )
    search_parser.add_argument(
        "query",
        help="Search query"
    )
    search_parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Maximum results to return"
    )

    # Parse arguments
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    # Route to appropriate function
    if args.command == "analyze":
        return analyze_project(args)
    elif args.command == "document":
        return document_model(args)
    elif args.command == "lineage":
        return analyze_lineage(args)
    elif args.command == "optimize":
        return optimize_sql(args)
    elif args.command == "search":
        return search_models(args)


if __name__ == "__main__":
    sys.exit(main())