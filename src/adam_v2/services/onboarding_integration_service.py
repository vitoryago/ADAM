"""
Onboarding Integration Service
Detects onboarding requests in chat and provides contextual responses
"""

import re
import json
from pathlib import Path
from typing import Dict, Any, Optional, List
import logging

from adam_v2.services.onboarding_service import OnboardingService, OnboardingPhase
from adam_v2.services.lineage_service import LineageService
from adam_v2.services.llm_service import LLMService

logger = logging.getLogger(__name__)

class OnboardingIntegrationService:
    """Integrates onboarding into the chat experience"""
    
    def __init__(self, llm_service: LLMService):
        self.llm_service = llm_service
        self.lineage_service = LineageService()
        self.onboarding_services: Dict[str, OnboardingService] = {}
        
        # Keywords that trigger onboarding detection
        self.onboarding_triggers = [
            "onboard", "onboarding", "get started", "learn about",
            "new to", "help me understand", "explain the project",
            "walk me through", "introduction to", "overview of",
            "how does .* work", "what is", "guide me",
            "teach me", "I'm new", "beginner", "tutorial"
        ]
        
        # Project area keywords
        self.area_keywords = {
            "marketing": ["marketing", "campaign", "analytics", "conversion", "funnel"],
            "data": ["data", "pipeline", "etl", "warehouse", "dbt", "transformation"],
            "frontend": ["frontend", "react", "ui", "interface", "component"],
            "backend": ["backend", "api", "server", "database", "service"],
            "ml": ["ml", "machine learning", "model", "training", "prediction"]
        }
    
    def is_onboarding_request(self, message: str) -> bool:
        """Check if a message is requesting onboarding"""
        message_lower = message.lower()
        
        for trigger in self.onboarding_triggers:
            if re.search(trigger, message_lower):
                return True
        
        return False
    
    def detect_focus_area(self, message: str) -> Optional[str]:
        """Detect which area of the project the user wants to focus on"""
        message_lower = message.lower()
        
        for area, keywords in self.area_keywords.items():
            for keyword in keywords:
                if keyword in message_lower:
                    return area
        
        return None
    
    def detect_user_level(self, message: str) -> str:
        """Detect user experience level from message context"""
        message_lower = message.lower()
        
        if any(word in message_lower for word in ["beginner", "new", "first time", "basic"]):
            return "beginner"
        elif any(word in message_lower for word in ["advanced", "expert", "deep dive", "complex"]):
            return "advanced"
        else:
            return "intermediate"
    
    async def process_onboarding_request(
        self,
        message: str,
        project_id: str,
        project_name: str,
        project_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """Process an onboarding request and return structured response"""
        
        # Detect parameters from message
        focus_area = self.detect_focus_area(message)
        user_level = self.detect_user_level(message)
        
        # Use current directory if no project path specified
        if not project_path:
            project_path = str(Path.cwd())
        
        # Get or create onboarding service for this project
        if project_id not in self.onboarding_services:
            self.onboarding_services[project_id] = OnboardingService(
                project_id=project_id,
                project_name=project_name,
                llm_service=self.llm_service
            )
        
        onboarding_service = self.onboarding_services[project_id]
        
        # Create onboarding path
        try:
            path = await onboarding_service.create_onboarding_path(
                project_path=project_path,
                user_level=user_level,
                focus_area=focus_area,
                custom_requirements=message
            )
            
            # Get first recommendation
            recommendation = await onboarding_service.get_next_recommendation(path.id)
            
            # Format response
            response = self._format_onboarding_response(path, recommendation, focus_area)
            
            return {
                "type": "onboarding",
                "path_id": path.id,
                "content": response,
                "metadata": {
                    "milestones": len(path.milestones),
                    "estimated_time": path.estimated_total_time,
                    "focus_area": focus_area,
                    "user_level": user_level
                }
            }
            
        except Exception as e:
            logger.error(f"Error creating onboarding path: {e}")
            return {
                "type": "error",
                "content": f"I encountered an issue creating your onboarding path. Error: {str(e)}",
                "metadata": {}
            }
    
    def _format_onboarding_response(self, path, recommendation, focus_area) -> str:
        """Format the onboarding response for chat"""
        
        focus_text = f" focusing on **{focus_area}**" if focus_area else ""
        
        response = f"""
## 🎓 Welcome to Your {path.project_name} Onboarding Journey!

I've created a personalized learning path for you{focus_text}. Here's your roadmap:

### 📊 Overview
- **Total Milestones:** {len(path.milestones)}
- **Estimated Time:** {path.estimated_total_time} minutes (~{path.estimated_total_time // 60} hours)
- **Experience Level:** {path.user_level}

### 🎯 Your Learning Path:
"""
        
        for i, milestone in enumerate(path.milestones[:3], 1):  # Show first 3 milestones
            emoji = "✅" if milestone.status.value == "completed" else "📍"
            response += f"\n{i}. {emoji} **{milestone.title}**"
            response += f"\n   - {milestone.description}"
            response += f"\n   - Time: {milestone.estimated_time} minutes"
            response += f"\n   - Tasks: {len(milestone.tasks)}"
            response += "\n"
        
        if len(path.milestones) > 3:
            response += f"\n... and {len(path.milestones) - 3} more milestones\n"
        
        response += f"""
### 🚀 First Step
{recommendation.get('message', 'Ready to begin your journey!')}

### 💡 Quick Actions
- Ask me **"What's next?"** to get your current task
- Say **"Show my progress"** to see how far you've come
- Request **"Explain [concept]"** for detailed explanations
- Type **"I completed [task]"** to mark progress

I'll guide you through each step, answer questions, and help you understand the codebase. Let's start with the first milestone!

**Ready to begin?** Just let me know when you want to start, or ask me any questions about the project.
"""
        
        return response
    
    async def handle_progress_update(
        self,
        message: str,
        project_id: str,
        path_id: str
    ) -> Dict[str, Any]:
        """Handle progress updates from chat"""
        
        if project_id not in self.onboarding_services:
            return {
                "type": "error",
                "content": "Onboarding session not found. Please start a new onboarding.",
                "metadata": {}
            }
        
        onboarding_service = self.onboarding_services[project_id]
        
        # Check for completion markers
        if "completed" in message.lower() or "done" in message.lower():
            # Try to identify which task was completed
            # This is simplified - in production, you'd want more sophisticated parsing
            
            if path_id in onboarding_service.paths:
                path = onboarding_service.paths[path_id]
                
                # Find current incomplete task
                for milestone in path.milestones:
                    if milestone.status.value != "completed":
                        for task in milestone.tasks:
                            if not task["completed"]:
                                # Mark this task as completed
                                await onboarding_service.update_milestone_progress(
                                    path_id=path_id,
                                    milestone_id=milestone.id,
                                    task_id=task["id"],
                                    completed=True
                                )
                                
                                # Get next recommendation
                                recommendation = await onboarding_service.get_next_recommendation(path_id)
                                
                                return {
                                    "type": "progress_update",
                                    "content": f"Great job! ✅ Task '{task['title']}' marked as complete.\n\n{recommendation['message']}",
                                    "metadata": {
                                        "progress": path.progress,
                                        "task_completed": task["title"]
                                    }
                                }
        
        # Handle other progress-related queries
        if "progress" in message.lower() or "status" in message.lower():
            if path_id in onboarding_service.paths:
                path = onboarding_service.paths[path_id]
                return {
                    "type": "progress_status",
                    "content": self._format_progress_status(path),
                    "metadata": {
                        "progress": path.progress
                    }
                }
        
        return {
            "type": "general",
            "content": None,
            "metadata": {}
        }
    
    def _format_progress_status(self, path) -> str:
        """Format current progress status"""
        
        completed_milestones = sum(1 for m in path.milestones if m.status.value == "completed")
        total_tasks = sum(len(m.tasks) for m in path.milestones)
        completed_tasks = sum(sum(1 for t in m.tasks if t["completed"]) for m in path.milestones)
        
        response = f"""
## 📊 Your Progress Update

**Overall Progress:** {path.progress:.1f}%

### Milestones
- Completed: {completed_milestones}/{len(path.milestones)}
- Current: {path.current_milestone or 'None'}

### Tasks
- Completed: {completed_tasks}/{total_tasks}

### Current Focus
"""
        
        for milestone in path.milestones:
            if milestone.id == path.current_milestone:
                response += f"**{milestone.title}**\n"
                for task in milestone.tasks[:3]:
                    status = "✅" if task["completed"] else "⏳"
                    response += f"  {status} {task['title']}\n"
                
                incomplete = [t for t in milestone.tasks if not t["completed"]]
                if incomplete:
                    response += f"\n**Next Task:** {incomplete[0]['title']}"
                
                break
        
        return response
    
    async def analyze_project_for_chat(
        self,
        project_path: str,
        message: str
    ) -> str:
        """Analyze project and provide insights for chat context"""
        
        # Run lineage analysis
        path = Path(project_path)
        result = self.lineage_service.analyze_directory(path)
        stats = self.lineage_service.get_statistics()
        
        # Build context for LLM
        context = f"""
        Project Analysis for {path.name}:
        - Files analyzed: {result['files_analyzed']}
        - Components: {result['nodes']}
        - Dependencies: {result['edges']}
        - Node types: {stats.get('node_types', {})}
        - Most connected components: {stats.get('most_connected', [])[:3]}
        
        User query: {message}
        """
        
        # Get AI insights
        prompt = f"""
        Based on this project analysis, provide helpful insights about the codebase 
        structure and suggest how the user can best learn about it. Keep response 
        concise and actionable.
        
        {context}
        """
        
        response = await self.llm_service.generate_response(prompt)
        
        return response