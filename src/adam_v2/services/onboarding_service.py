"""
ADAM Onboarding Service
Provides intelligent project onboarding with context-aware guidance
"""

import asyncio
import json
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import logging
from datetime import datetime

from adam_v2.services.lineage_service import LineageService, LineageAnalyzer
from adam_v2.services.memory_service import ProjectMemoryService
from adam_v2.services.llm_service import LLMService
from adam_v2.services.dbt_knowledge_service import DBTKnowledgeService

logger = logging.getLogger(__name__)

class OnboardingPhase(Enum):
    """Phases of the onboarding process"""
    DISCOVERY = "discovery"
    ANALYSIS = "analysis"
    PLANNING = "planning"
    LEARNING = "learning"
    PRACTICE = "practice"
    MASTERY = "mastery"

class MilestoneStatus(Enum):
    """Status of onboarding milestones"""
    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    SKIPPED = "skipped"

@dataclass
class OnboardingMilestone:
    """Represents a milestone in the onboarding journey"""
    id: str
    title: str
    description: str
    phase: OnboardingPhase
    status: MilestoneStatus
    tasks: List[Dict[str, Any]]
    dependencies: List[str]
    estimated_time: int  # in minutes
    resources: List[Dict[str, str]]
    completion_criteria: List[str]
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

@dataclass
class OnboardingPath:
    """Represents a complete onboarding path"""
    id: str
    project_name: str
    user_level: str  # beginner, intermediate, advanced
    focus_area: Optional[str]  # e.g., "marketing", "data engineering"
    milestones: List[OnboardingMilestone]
    current_milestone: Optional[str]
    progress: float
    estimated_total_time: int
    created_at: datetime
    updated_at: datetime

class OnboardingService:
    """Service for intelligent project onboarding"""
    
    def __init__(self, project_id: str, project_name: str, llm_service: LLMService):
        self.memory_service = ProjectMemoryService(project_id, project_name)
        self.llm_service = llm_service
        self.lineage_service = LineageService()
        self.dbt_knowledge = DBTKnowledgeService()
        self.paths: Dict[str, OnboardingPath] = {}
        
    async def create_onboarding_path(
        self, 
        project_path: str,
        user_level: str = "beginner",
        focus_area: Optional[str] = None,
        custom_requirements: Optional[str] = None
    ) -> OnboardingPath:
        """Create a personalized onboarding path based on project analysis"""
        
        logger.info(f"Creating onboarding path for {project_path}, level: {user_level}")
        
        # Analyze project structure and lineage
        project_analysis = await self._analyze_project(project_path, focus_area)
        
        # Generate context-aware milestones
        milestones = await self._generate_milestones(
            project_analysis, 
            user_level, 
            focus_area,
            custom_requirements
        )
        
        # Create onboarding path
        path = OnboardingPath(
            id=f"onboarding_{datetime.now().timestamp()}",
            project_name=Path(project_path).name,
            user_level=user_level,
            focus_area=focus_area,
            milestones=milestones,
            current_milestone=milestones[0].id if milestones else None,
            progress=0.0,
            estimated_total_time=sum(m.estimated_time for m in milestones),
            created_at=datetime.now(),
            updated_at=datetime.now()
        )
        
        # Store in memory for persistence (disabled for now due to ChromaDB metadata issues)
        # await self._save_to_memory(path, project_analysis)
        
        self.paths[path.id] = path
        return path
    
    async def _analyze_project(self, project_path: str, focus_area: Optional[str]) -> Dict[str, Any]:
        """Analyze project structure and create comprehensive context"""
        
        path = Path(project_path)
        
        # Determine what to analyze
        if focus_area:
            analyze_path = path / focus_area if (path / focus_area).exists() else path
        else:
            analyze_path = path
            
        # Run lineage analysis
        lineage_result = self.lineage_service.analyze_directory(analyze_path)
        
        # Get project statistics
        stats = self.lineage_service.get_statistics()
        
        # Identify key components
        key_components = self._identify_key_components(stats)
        
        # Check for specific patterns (DBT, APIs, etc.)
        project_patterns = self._detect_project_patterns(analyze_path)
        
        # Create comprehensive analysis
        analysis = {
            "path": str(project_path),
            "focus_area": focus_area,
            "lineage": lineage_result,
            "statistics": stats,
            "key_components": key_components,
            "patterns": project_patterns,
            "complexity_score": self._calculate_complexity_score(stats),
            "recommended_learning_time": self._estimate_learning_time(stats)
        }
        
        return analysis
    
    def _identify_key_components(self, stats: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identify the most important components to learn"""
        
        components = []
        
        # Get most connected nodes (likely important)
        if "most_connected" in stats:
            for node_id, connections in stats["most_connected"][:5]:
                if node_id in self.lineage_service.analyzer.nodes:
                    node = self.lineage_service.analyzer.nodes[node_id]
                    components.append({
                        "id": node.id,
                        "name": node.name,
                        "type": node.type.value,
                        "importance": "high",
                        "connections": connections
                    })
                    
        return components
    
    def _detect_project_patterns(self, path: Path) -> Dict[str, bool]:
        """Detect specific project patterns and technologies"""
        
        patterns = {
            "has_dbt": any(path.rglob("dbt_project.yml")),
            "has_sql": any(path.rglob("*.sql")),
            "has_python": any(path.rglob("*.py")),
            "has_api": any(path.rglob("*api*.py")) or any(path.rglob("*route*.py")),
            "has_tests": (path / "tests").exists() or (path / "test").exists(),
            "has_docker": (path / "Dockerfile").exists() or (path / "docker-compose.yml").exists(),
            "has_ci_cd": any(path.rglob(".github/workflows/*")) or (path / ".gitlab-ci.yml").exists(),
            "has_documentation": (path / "docs").exists() or (path / "README.md").exists()
        }
        
        return patterns
    
    def _calculate_complexity_score(self, stats: Dict[str, Any]) -> float:
        """Calculate project complexity score (0-100)"""
        
        score = 0.0
        
        # Factor in number of nodes and edges
        score += min(stats.get("total_nodes", 0) / 10, 30)
        score += min(stats.get("total_edges", 0) / 20, 30)
        
        # Factor in component diversity
        node_types = stats.get("node_types", {})
        score += min(len(node_types) * 5, 20)
        
        # Factor in connectivity
        avg_degree = stats.get("average_degree", 0)
        score += min(avg_degree * 2, 20)
        
        return min(score, 100)
    
    def _estimate_learning_time(self, stats: Dict[str, Any]) -> int:
        """Estimate learning time in hours based on complexity"""
        
        complexity = self._calculate_complexity_score(stats)
        
        # Base time + complexity factor
        base_time = 4  # hours
        complexity_time = (complexity / 100) * 16  # up to 16 additional hours
        
        return int(base_time + complexity_time)
    
    async def _generate_milestones(
        self,
        analysis: Dict[str, Any],
        user_level: str,
        focus_area: Optional[str],
        custom_requirements: Optional[str]
    ) -> List[OnboardingMilestone]:
        """Generate intelligent milestones based on project analysis"""
        
        milestones = []
        
        # Phase 1: Discovery
        milestones.append(OnboardingMilestone(
            id="milestone_1",
            title="Project Overview & Architecture",
            description="Understand the overall project structure and architecture",
            phase=OnboardingPhase.DISCOVERY,
            status=MilestoneStatus.NOT_STARTED,
            tasks=[
                {"id": "1.1", "title": "Review project documentation", "completed": False},
                {"id": "1.2", "title": "Explore directory structure", "completed": False},
                {"id": "1.3", "title": "Identify main components", "completed": False},
                {"id": "1.4", "title": "Understand data flow", "completed": False}
            ],
            dependencies=[],
            estimated_time=60,
            resources=[
                {"type": "documentation", "path": "README.md"},
                {"type": "diagram", "path": "lineage_graph"}
            ],
            completion_criteria=[
                "Can explain project purpose",
                "Can identify main components",
                "Understands basic data flow"
            ]
        ))
        
        # Phase 2: Analysis - Key Components
        if analysis["key_components"]:
            milestones.append(OnboardingMilestone(
                id="milestone_2",
                title="Understanding Key Components",
                description="Deep dive into the most important parts of the system",
                phase=OnboardingPhase.ANALYSIS,
                status=MilestoneStatus.NOT_STARTED,
                tasks=[
                    {
                        "id": f"2.{i+1}", 
                        "title": f"Study {comp['name']} ({comp['type']})",
                        "completed": False
                    }
                    for i, comp in enumerate(analysis["key_components"][:5])
                ],
                dependencies=["milestone_1"],
                estimated_time=120,
                resources=[
                    {"type": "code", "path": comp["id"]} 
                    for comp in analysis["key_components"][:3]
                ],
                completion_criteria=[
                    "Can explain each component's purpose",
                    "Understands component interactions",
                    "Can modify basic functionality"
                ]
            ))
        
        # Phase 3: Technology-specific milestones
        patterns = analysis["patterns"]
        
        if patterns.get("has_dbt"):
            milestones.append(OnboardingMilestone(
                id="milestone_dbt",
                title="Master DBT Models & Transformations",
                description="Learn the data transformation layer",
                phase=OnboardingPhase.LEARNING,
                status=MilestoneStatus.NOT_STARTED,
                tasks=[
                    {"id": "dbt.1", "title": "Understand DBT project structure", "completed": False},
                    {"id": "dbt.2", "title": "Review staging models", "completed": False},
                    {"id": "dbt.3", "title": "Study transformation logic", "completed": False},
                    {"id": "dbt.4", "title": "Run and test models", "completed": False}
                ],
                dependencies=["milestone_2"],
                estimated_time=180,
                resources=[
                    {"type": "documentation", "path": "dbt_project.yml"},
                    {"type": "models", "path": "models/"}
                ],
                completion_criteria=[
                    "Can run DBT models",
                    "Understands model dependencies",
                    "Can create new transformations"
                ]
            ))
        
        if patterns.get("has_api"):
            milestones.append(OnboardingMilestone(
                id="milestone_api",
                title="API Architecture & Endpoints",
                description="Understand the API layer and services",
                phase=OnboardingPhase.LEARNING,
                status=MilestoneStatus.NOT_STARTED,
                tasks=[
                    {"id": "api.1", "title": "Review API documentation", "completed": False},
                    {"id": "api.2", "title": "Test key endpoints", "completed": False},
                    {"id": "api.3", "title": "Understand authentication", "completed": False},
                    {"id": "api.4", "title": "Study data contracts", "completed": False}
                ],
                dependencies=["milestone_2"],
                estimated_time=90,
                resources=[
                    {"type": "api_docs", "path": "/api/docs"},
                    {"type": "postman", "path": "api_collection.json"}
                ],
                completion_criteria=[
                    "Can call all major endpoints",
                    "Understands request/response formats",
                    "Can debug API issues"
                ]
            ))
        
        # Phase 4: Practice
        milestones.append(OnboardingMilestone(
            id="milestone_practice",
            title="Hands-on Practice Tasks",
            description="Apply your knowledge with real tasks",
            phase=OnboardingPhase.PRACTICE,
            status=MilestoneStatus.NOT_STARTED,
            tasks=[
                {"id": "p.1", "title": "Make a simple code change", "completed": False},
                {"id": "p.2", "title": "Add a test case", "completed": False},
                {"id": "p.3", "title": "Fix a small bug", "completed": False},
                {"id": "p.4", "title": "Add documentation", "completed": False}
            ],
            dependencies=[m.id for m in milestones if m.phase == OnboardingPhase.LEARNING],
            estimated_time=240,
            resources=[
                {"type": "issues", "path": "github_issues"},
                {"type": "guide", "path": "contribution_guide.md"}
            ],
            completion_criteria=[
                "Successfully merged a PR",
                "All tests passing",
                "Code review approved"
            ]
        ))
        
        # Adjust based on user level
        if user_level == "advanced":
            # Add architecture and optimization milestones
            milestones.append(OnboardingMilestone(
                id="milestone_advanced",
                title="Architecture & Optimization",
                description="Deep architectural understanding and optimization",
                phase=OnboardingPhase.MASTERY,
                status=MilestoneStatus.NOT_STARTED,
                tasks=[
                    {"id": "a.1", "title": "Analyze performance bottlenecks", "completed": False},
                    {"id": "a.2", "title": "Propose architecture improvements", "completed": False},
                    {"id": "a.3", "title": "Implement optimization", "completed": False}
                ],
                dependencies=["milestone_practice"],
                estimated_time=360,
                resources=[
                    {"type": "metrics", "path": "performance_metrics"},
                    {"type": "architecture", "path": "architecture_docs"}
                ],
                completion_criteria=[
                    "Identified optimization opportunities",
                    "Implemented measurable improvements",
                    "Documented architectural decisions"
                ]
            ))
        
        return milestones
    
    async def _save_to_memory(self, path: OnboardingPath, analysis: Dict[str, Any]):
        """Save onboarding path and analysis to memory"""
        
        # Create memory entry
        memory_content = f"""
        Onboarding Path Created for {path.project_name}
        User Level: {path.user_level}
        Focus Area: {path.focus_area or 'Full Project'}
        Total Milestones: {len(path.milestones)}
        Estimated Time: {path.estimated_total_time} minutes
        
        Project Analysis:
        - Complexity Score: {analysis.get('complexity_score', 0):.1f}/100
        - Files Analyzed: {analysis['lineage'].get('files_analyzed', 0)}
        - Components Found: {analysis['lineage'].get('nodes', 0)}
        - Key Technologies: {', '.join([k for k, v in analysis['patterns'].items() if v])}
        
        Learning Path:
        {self._format_milestone_summary(path.milestones)}
        """
        
        await self.memory_service.store_memory(
            content=memory_content,
            memory_type="knowledge",
            metadata={
                "type": "onboarding_path",
                "path_id": path.id,
                "project": path.project_name,
                "user_level": path.user_level
            }
        )
    
    def _format_milestone_summary(self, milestones: List[OnboardingMilestone]) -> str:
        """Format milestones for memory storage"""
        
        lines = []
        for i, milestone in enumerate(milestones, 1):
            lines.append(f"{i}. {milestone.title} ({milestone.phase.value})")
            lines.append(f"   Time: {milestone.estimated_time} min")
            lines.append(f"   Tasks: {len(milestone.tasks)}")
            
        return "\n".join(lines)
    
    async def update_milestone_progress(
        self, 
        path_id: str, 
        milestone_id: str, 
        task_id: str,
        completed: bool
    ) -> OnboardingPath:
        """Update progress on a specific task"""
        
        if path_id not in self.paths:
            raise ValueError(f"Path {path_id} not found")
            
        path = self.paths[path_id]
        
        # Find and update the task
        for milestone in path.milestones:
            if milestone.id == milestone_id:
                for task in milestone.tasks:
                    if task["id"] == task_id:
                        task["completed"] = completed
                        break
                        
                # Update milestone status
                completed_tasks = sum(1 for t in milestone.tasks if t["completed"])
                if completed_tasks == 0:
                    milestone.status = MilestoneStatus.NOT_STARTED
                elif completed_tasks < len(milestone.tasks):
                    milestone.status = MilestoneStatus.IN_PROGRESS
                else:
                    milestone.status = MilestoneStatus.COMPLETED
                    
        # Update overall progress
        total_tasks = sum(len(m.tasks) for m in path.milestones)
        completed_tasks = sum(
            sum(1 for t in m.tasks if t["completed"]) 
            for m in path.milestones
        )
        path.progress = (completed_tasks / total_tasks * 100) if total_tasks > 0 else 0
        
        path.updated_at = datetime.now()
        
        return path
    
    async def get_next_recommendation(self, path_id: str) -> Dict[str, Any]:
        """Get AI-powered recommendation for next step"""
        
        if path_id not in self.paths:
            raise ValueError(f"Path {path_id} not found")
            
        path = self.paths[path_id]
        
        # Find current milestone and next task
        current_milestone = None
        next_task = None
        
        for milestone in path.milestones:
            if milestone.status != MilestoneStatus.COMPLETED:
                current_milestone = milestone
                for task in milestone.tasks:
                    if not task["completed"]:
                        next_task = task
                        break
                break
                
        if not current_milestone or not next_task:
            return {
                "message": "Congratulations! You've completed all milestones!",
                "milestone": None,
                "task": None
            }
            
        # Get AI recommendation
        prompt = f"""
        The user is onboarding to a {path.project_name} project.
        Current milestone: {current_milestone.title}
        Next task: {next_task['title']}
        
        Provide a helpful, encouraging message with specific guidance for completing this task.
        Include any tips or resources that might help.
        Keep it concise and actionable.
        """
        
        recommendation = await self.llm_service.generate_response(prompt)
        
        return {
            "message": recommendation,
            "milestone": asdict(current_milestone),
            "task": next_task,
            "progress": path.progress
        }
    
    def export_path(self, path_id: str, format: str = "json") -> str:
        """Export onboarding path in various formats"""
        
        if path_id not in self.paths:
            raise ValueError(f"Path {path_id} not found")
            
        path = self.paths[path_id]
        
        if format == "json":
            return json.dumps(asdict(path), indent=2, default=str)
            
        elif format == "markdown":
            lines = [
                f"# Onboarding Path: {path.project_name}",
                f"",
                f"**User Level:** {path.user_level}",
                f"**Progress:** {path.progress:.1f}%",
                f"**Total Time:** {path.estimated_total_time} minutes",
                f"",
                f"## Milestones",
                f""
            ]
            
            for milestone in path.milestones:
                status_icon = "✅" if milestone.status == MilestoneStatus.COMPLETED else "⏳"
                lines.append(f"### {status_icon} {milestone.title}")
                lines.append(f"*{milestone.description}*")
                lines.append(f"")
                lines.append(f"**Tasks:**")
                for task in milestone.tasks:
                    check = "☑️" if task["completed"] else "☐"
                    lines.append(f"- {check} {task['title']}")
                lines.append(f"")
                
            return "\n".join(lines)
            
        else:
            raise ValueError(f"Unknown format: {format}")