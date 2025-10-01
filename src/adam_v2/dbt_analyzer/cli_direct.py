#!/usr/bin/env python3
"""
ADAM DBT Direct Reader CLI - No manifest.json needed!
Works directly with your DBT source files.
"""
import argparse
import sys
import json
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich.tree import Tree
from rich import print as rprint

from .direct_reader import DirectDBTReader

console = Console()


def analyze_project(args):
    """Analyze DBT project from source files"""
    console.print(f"\n🔍 Analyzing DBT project at: [cyan]{args.project_path}[/cyan]\n")

    reader = DirectDBTReader(args.project_path)
    if not reader.read_project():
        console.print("❌ [red]Failed to read project[/red]")
        return 1

    stats = reader.get_project_stats()

    # Display stats
    console.print(f"📦 [bold]Project:[/bold] {stats['project_name']}")
    console.print(f"📊 [bold]Models:[/bold] {stats['total_models']}")
    console.print(f"📁 [bold]Sources:[/bold] {stats['total_sources']}")
    console.print(f"🔗 [bold]Refs:[/bold] {stats['total_refs']}")
    console.print(f"📌 [bold]Source Refs:[/bold] {stats['total_source_refs']}\n")

    # Complexity distribution
    console.print("[bold]Complexity Distribution:[/bold]")
    console.print(f"  Simple (< 5): {stats['complexity']['simple']}")
    console.print(f"  Medium (5-15): {stats['complexity']['medium']}")
    console.print(f"  Complex (> 15): {stats['complexity']['complex']}\n")

    # Materialization
    if stats['materializations']:
        console.print("[bold]Materializations:[/bold]")
        for mat, count in stats['materializations'].items():
            console.print(f"  {mat}: {count}")

    # List all models with complexity
    if args.verbose:
        console.print("\n[bold]All Models:[/bold]")
        table = Table(show_header=True)
        table.add_column("Model", style="cyan")
        table.add_column("Complexity", justify="right")
        table.add_column("Refs", justify="right")
        table.add_column("Sources", justify="right")
        table.add_column("Materialization")

        for name, model in sorted(reader.models.items()):
            table.add_row(
                name,
                str(model.complexity_score),
                str(len(model.refs)),
                str(len(model.sources)),
                model.materialization or "view"
            )

        console.print(table)

    return 0


def show_model(args):
    """Show detailed info about a specific model"""
    reader = DirectDBTReader(args.project_path)
    if not reader.read_project():
        console.print("❌ [red]Failed to read project[/red]")
        return 1

    model = reader.get_model(args.model_name)
    if not model:
        console.print(f"❌ [red]Model '{args.model_name}' not found[/red]")
        return 1

    # Header
    console.print(f"\n📄 [bold cyan]{model.name}[/bold cyan]")
    console.print(f"📁 Path: {model.path}")
    if model.description:
        console.print(f"📝 {model.description}\n")

    # Dependencies
    if model.refs or model.sources:
        console.print("[bold]Dependencies:[/bold]")
        if model.refs:
            console.print(f"  Models: {', '.join(model.refs)}")
        if model.sources:
            sources_str = [f"{s[0]}.{s[1]}" for s in model.sources]
            console.print(f"  Sources: {', '.join(sources_str)}")
        console.print()

    # Downstream
    downstream = reader.get_downstream_models(model.name)
    if downstream:
        console.print(f"[bold]Used by:[/bold] {', '.join(downstream)}\n")

    # Characteristics
    console.print("[bold]Characteristics:[/bold]")
    console.print(f"  Complexity Score: {model.complexity_score}")
    console.print(f"  CTEs: {len(model.ctes)}")
    console.print(f"  Window Functions: {'✓' if model.uses_window_functions else '✗'}")
    console.print(f"  Joins: {'✓' if model.uses_joins else '✗'}")
    console.print(f"  Aggregations: {'✓' if model.uses_aggregations else '✗'}")
    console.print(f"  Materialization: {model.materialization or 'view'}\n")

    # Tests
    if model.tests:
        console.print(f"[bold]Tests ({len(model.tests)}):[/bold]")
        for test in model.tests:
            console.print(f"  • {test}")
        console.print()

    # Columns
    if model.columns:
        console.print("[bold]Columns:[/bold]")
        for col_name, col_desc in model.columns.items():
            console.print(f"  • [cyan]{col_name}[/cyan]: {col_desc or '(no description)'}")
        console.print()

    # SQL preview (first 20 lines)
    if args.show_sql:
        console.print("[bold]SQL Preview:[/bold]")
        lines = model.sql.split('\n')[:20]
        for line in lines:
            console.print(f"  {line}")
        if len(model.sql.split('\n')) > 20:
            console.print("  ...")
        console.print()

    return 0


def show_lineage(args):
    """Show lineage tree for a model"""
    reader = DirectDBTReader(args.project_path)
    if not reader.read_project():
        console.print("❌ [red]Failed to read project[/red]")
        return 1

    model = reader.get_model(args.model_name)
    if not model:
        console.print(f"❌ [red]Model '{args.model_name}' not found[/red]")
        return 1

    # Build tree
    tree = Tree(f"[bold cyan]{model.name}[/bold cyan]")

    # Upstream
    if model.refs or model.sources:
        upstream_branch = tree.add("[yellow]⬆ Upstream[/yellow]")

        for ref in model.refs:
            ref_model = reader.get_model(ref)
            if ref_model:
                ref_branch = upstream_branch.add(f"[cyan]{ref}[/cyan] (complexity: {ref_model.complexity_score})")
                # Show second level
                for nested_ref in ref_model.refs[:3]:  # Limit to 3
                    ref_branch.add(f"[dim]{nested_ref}[/dim]")
                if len(ref_model.refs) > 3:
                    ref_branch.add("[dim]...[/dim]")

        for source_schema, source_table in model.sources:
            upstream_branch.add(f"[green]source: {source_schema}.{source_table}[/green]")

    # Downstream
    downstream = reader.get_downstream_models(model.name)
    if downstream:
        downstream_branch = tree.add("[yellow]⬇ Downstream[/yellow]")
        for down_name in downstream:
            down_model = reader.get_model(down_name)
            downstream_branch.add(f"[cyan]{down_name}[/cyan] (complexity: {down_model.complexity_score})")

    console.print("\n")
    console.print(tree)
    console.print()

    return 0


def list_models(args):
    """List all models with filters"""
    reader = DirectDBTReader(args.project_path)
    if not reader.read_project():
        console.print("❌ [red]Failed to read project[/red]")
        return 1

    models = list(reader.models.values())

    # Apply filters
    if args.complex:
        models = [m for m in models if m.complexity_score >= 15]
    if args.simple:
        models = [m for m in models if m.complexity_score < 5]
    if args.has_tests:
        models = [m for m in models if m.tests]

    # Sort
    if args.sort == "complexity":
        models.sort(key=lambda m: m.complexity_score, reverse=True)
    else:
        models.sort(key=lambda m: m.name)

    # Display
    table = Table(show_header=True, title=f"Models ({len(models)})")
    table.add_column("Model", style="cyan")
    table.add_column("Complexity", justify="right")
    table.add_column("Refs", justify="right")
    table.add_column("Tests", justify="right")
    table.add_column("Materialization")

    for model in models:
        table.add_row(
            model.name,
            str(model.complexity_score),
            str(len(model.refs)),
            str(len(model.tests)),
            model.materialization or "view"
        )

    console.print("\n")
    console.print(table)
    console.print()

    return 0


def search_models(args):
    """Search models by name or SQL content"""
    reader = DirectDBTReader(args.project_path)
    if not reader.read_project():
        console.print("❌ [red]Failed to read project[/red]")
        return 1

    query = args.query.lower()
    results = []

    for name, model in reader.models.items():
        # Search in name
        if query in name.lower():
            results.append((name, "name", name))
            continue

        # Search in SQL
        if query in model.sql.lower():
            # Find context around match
            lines = model.sql.split('\n')
            for i, line in enumerate(lines):
                if query in line.lower():
                    results.append((name, "sql", line.strip()[:80]))
                    break

    if not results:
        console.print(f"❌ [red]No results found for '{args.query}'[/red]")
        return 1

    console.print(f"\n🔍 Found {len(results)} results for '[cyan]{args.query}[/cyan]':\n")

    for model_name, match_type, context in results:
        if match_type == "name":
            console.print(f"  📄 [cyan]{model_name}[/cyan]")
        else:
            console.print(f"  📄 [cyan]{model_name}[/cyan]: {context}...")

    console.print()
    return 0


def export_json(args):
    """Export project structure to JSON"""
    reader = DirectDBTReader(args.project_path)
    if not reader.read_project():
        console.print("❌ [red]Failed to read project[/red]")
        return 1

    output = {
        "stats": reader.get_project_stats(),
        "models": {}
    }

    for name, model in reader.models.items():
        output["models"][name] = {
            "path": model.path,
            "refs": model.refs,
            "sources": [[s[0], s[1]] for s in model.sources],
            "complexity": model.complexity_score,
            "materialization": model.materialization,
            "tests": model.tests,
            "columns": model.columns
        }

    output_file = Path(args.output) if args.output else Path("dbt_project.json")
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)

    console.print(f"✅ [green]Exported to {output_file}[/green]")
    return 0


def main():
    parser = argparse.ArgumentParser(
        description="ADAM DBT Direct Reader - Analyze DBT projects without manifest.json"
    )
    parser.add_argument("--project-path", "-p", default=".", help="Path to DBT project")

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Analyze command
    analyze_parser = subparsers.add_parser("analyze", help="Analyze entire project")
    analyze_parser.add_argument("--verbose", "-v", action="store_true", help="Show all models")

    # Show model command
    show_parser = subparsers.add_parser("show", help="Show model details")
    show_parser.add_argument("model_name", help="Name of the model")
    show_parser.add_argument("--show-sql", "-s", action="store_true", help="Show SQL preview")

    # Lineage command
    lineage_parser = subparsers.add_parser("lineage", help="Show model lineage")
    lineage_parser.add_argument("model_name", help="Name of the model")

    # List command
    list_parser = subparsers.add_parser("list", help="List all models")
    list_parser.add_argument("--complex", action="store_true", help="Show only complex models")
    list_parser.add_argument("--simple", action="store_true", help="Show only simple models")
    list_parser.add_argument("--has-tests", action="store_true", help="Show only models with tests")
    list_parser.add_argument("--sort", choices=["name", "complexity"], default="name")

    # Search command
    search_parser = subparsers.add_parser("search", help="Search models")
    search_parser.add_argument("query", help="Search query")

    # Export command
    export_parser = subparsers.add_parser("export", help="Export to JSON")
    export_parser.add_argument("--output", "-o", help="Output file path")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    # Route to appropriate function
    commands = {
        "analyze": analyze_project,
        "show": show_model,
        "lineage": show_lineage,
        "list": list_models,
        "search": search_models,
        "export": export_json
    }

    return commands[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
