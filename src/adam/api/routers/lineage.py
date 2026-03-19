"""
Lineage API endpoints for ADAM 4.0
"""

from fastapi import APIRouter, HTTPException, Query, Depends
from pathlib import Path
from typing import Dict, Any, Optional, List
import logging

router = APIRouter()
logger = logging.getLogger(__name__)

# Lazy-loaded lineage service
_lineage_service = None


def _get_lineage_service():
    """Lazy-load LineageService to avoid import-time failures."""
    global _lineage_service
    if _lineage_service is None:
        from adam.knowledge.lineage_service import LineageService
        _lineage_service = LineageService()
    return _lineage_service


@router.post("/analyze")
async def analyze_project(
    path: str,
    pattern: str = "*",
    focus_area: Optional[str] = None
) -> Dict[str, Any]:
    """
    Analyze a project directory and build lineage graph
    """
    try:
        lineage_service = _get_lineage_service()
        project_path = Path(path)

        if focus_area:
            project_path = project_path / focus_area

        if not project_path.exists():
            raise HTTPException(status_code=404, detail=f"Path not found: {project_path}")

        result = lineage_service.analyze_directory(project_path, pattern)

        return {
            "status": "success",
            "path": str(project_path),
            "result": result
        }

    except Exception as e:
        logger.error(f"Error analyzing project: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/node/{node_id}")
async def get_node_lineage(
    node_id: str,
    direction: str = Query("both", pattern="^(upstream|downstream|both)$"),
    depth: int = Query(3, ge=1, le=10)
) -> Dict[str, Any]:
    """Get lineage for a specific node"""
    try:
        lineage_service = _get_lineage_service()
        lineage = lineage_service.get_lineage(node_id, direction, depth)

        if "error" in lineage:
            raise HTTPException(status_code=404, detail=lineage["error"])

        return lineage

    except Exception as e:
        logger.error(f"Error getting lineage: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/impact/{node_id}")
async def analyze_impact(node_id: str) -> Dict[str, Any]:
    """Analyze the impact of changes to a node"""
    try:
        lineage_service = _get_lineage_service()
        impact = lineage_service.get_impact_analysis(node_id)

        if "error" in impact:
            raise HTTPException(status_code=404, detail=impact["error"])

        return impact

    except Exception as e:
        logger.error(f"Error analyzing impact: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/statistics")
async def get_statistics() -> Dict[str, Any]:
    """Get statistics about the current lineage graph"""
    try:
        lineage_service = _get_lineage_service()
        return lineage_service.get_statistics()
    except Exception as e:
        logger.error(f"Error getting statistics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/export")
async def export_lineage(
    format: str = Query("json", pattern="^(json|dot|mermaid)$")
) -> Any:
    """Export lineage graph in various formats"""
    try:
        lineage_service = _get_lineage_service()
        result = lineage_service.export_lineage(format)

        if format == "json":
            import json
            return json.loads(result)
        else:
            return {"format": format, "content": result}

    except Exception as e:
        logger.error(f"Error exporting lineage: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/search")
async def search_lineage(
    query: str,
    node_type: Optional[str] = None,
    limit: int = Query(10, ge=1, le=100)
) -> List[Dict[str, Any]]:
    """Search for nodes in the lineage graph"""
    try:
        lineage_service = _get_lineage_service()
        results = []

        for node_id, node in lineage_service.analyzer.nodes.items():
            if node_type and node.type.value != node_type:
                continue

            if query.lower() in node.name.lower():
                results.append({
                    "id": node.id,
                    "name": node.name,
                    "type": node.type.value,
                    "path": node.path,
                    "metadata": node.metadata
                })

            if len(results) >= limit:
                break

        return results

    except Exception as e:
        logger.error(f"Error searching lineage: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/graph/visualization")
async def get_visualization_data() -> Dict[str, Any]:
    """Get graph data formatted for visualization"""
    try:
        lineage_service = _get_lineage_service()
        nodes = []
        edges = []

        for node_id, node in lineage_service.analyzer.nodes.items():
            nodes.append({
                "id": node_id,
                "label": node.name,
                "type": node.type.value,
                "metadata": node.metadata
            })

        for edge in lineage_service.analyzer.edges:
            edges.append({
                "source": edge.source_id,
                "target": edge.target_id,
                "relationship": edge.relationship.value,
                "metadata": edge.metadata
            })

        return {
            "nodes": nodes,
            "edges": edges,
            "statistics": lineage_service.get_statistics()
        }

    except Exception as e:
        logger.error(f"Error getting visualization data: {e}")
        raise HTTPException(status_code=500, detail=str(e))
