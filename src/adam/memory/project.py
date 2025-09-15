"""
Project-Aware Memory System for ADAM
Extends the base memory system with project isolation
"""
from typing import Optional, Dict, Any, List
from datetime import datetime
import logging

from .core import ADAMMemoryAdvanced, Memory, MemoryType
from ..project_manager import ProjectManager, Project
from ..screen_capture import ScreenCaptureService

logger = logging.getLogger(__name__)


class ProjectAwareMemory(ADAMMemoryAdvanced):
    """
    Enhanced memory system that isolates memories by project.
    Each project has its own ChromaDB collection and can store screen captures.
    """
    
    def __init__(self, 
                 persist_directory: str = "./data/adam_memory",
                 embedding_model: str = "all-mpnet-base-v2",
                 project_manager: Optional[ProjectManager] = None):
        """
        Initialize project-aware memory system.
        
        Args:
            persist_directory: Base directory for memory storage
            embedding_model: Model for embeddings
            project_manager: Project manager instance
        """
        # Initialize project manager
        self.project_manager = project_manager or ProjectManager()
        
        # Get active project or create default
        self.current_project = self.project_manager.get_active_project()
        if not self.current_project:
            self.current_project = self.project_manager.get_or_create_default()
        
        # Initialize base memory with project-specific collection
        super().__init__(
            persist_directory=persist_directory,
            embedding_model=embedding_model,
            collection_name=f"adam_memory_{embedding_model}_{self.current_project.id}"
        )
        
        # Initialize screen capture service
        self.screen_capture = ScreenCaptureService(
            storage_path=f"{persist_directory}/screen_captures/{self.current_project.id}"
        )
        
        logger.info(f"Initialized project-aware memory for project: {self.current_project.name}")
    
    def switch_project(self, project_id: str) -> bool:
        """
        Switch to a different project's memory space.
        
        Args:
            project_id: ID of the project to switch to
            
        Returns:
            True if switch was successful
        """
        if self.project_manager.set_active_project(project_id):
            self.current_project = self.project_manager.get_project(project_id)
            
            # Reinitialize ChromaDB collection for new project
            self.collection_name = f"adam_memory_{self.embedding_model_name}_{project_id}"
            self._initialize_collection()
            
            # Update screen capture path
            self.screen_capture.storage_path = Path(
                f"{self.persist_directory}/screen_captures/{project_id}"
            )
            self.screen_capture.storage_path.mkdir(parents=True, exist_ok=True)
            
            logger.info(f"Switched to project: {self.current_project.name}")
            return True
        
        return False
    
    def create_project(self, name: str, description: str = "") -> Project:
        """Create a new project and switch to it"""
        project = self.project_manager.create_project(name, description)
        self.switch_project(project.id)
        return project
    
    def list_projects(self) -> List[Project]:
        """List all available projects"""
        return self.project_manager.list_projects()
    
    def remember_with_screen(self, 
                           query: str, 
                           response: str,
                           screen_capture: Optional[bytes] = None,
                           capture_screen_now: bool = False,
                           **kwargs) -> Optional[str]:
        """
        Remember something with optional screen context.
        
        Args:
            query: User query
            response: AI response
            screen_capture: Pre-captured screen data
            capture_screen_now: Whether to capture screen now
            **kwargs: Additional arguments for remember_if_worthy
            
        Returns:
            Memory ID if stored
        """
        # Capture screen if requested
        if capture_screen_now and not screen_capture:
            screen_capture = self.screen_capture.capture_screen()
        
        # Add screen context if available
        context = kwargs.get('context', {})
        if screen_capture:
            # Save screen capture
            filename = self.screen_capture.save_capture(
                screen_capture, 
                context=f"Query: {query}"
            )
            
            # Extract text from screen
            screen_text = self.screen_capture.extract_text_from_image(screen_capture)
            
            # Update context
            context.update({
                "screen_capture": filename,
                "screen_content": screen_text[:1000] if screen_text else None,
                "has_screen": True
            })
            
            # Force memory type to screen analysis
            kwargs['memory_type'] = MemoryType.SCREEN_ANALYSIS
        
        kwargs['context'] = context
        
        # Store memory
        memory_id = self.remember_if_worthy(query, response, **kwargs)
        
        # Update project stats
        if memory_id:
            stats = {
                "total_memories": len(self.collection.get()['ids']),
                "screen_captures": 1 if screen_capture else 0
            }
            self.project_manager.update_project_stats(self.current_project.id, stats)
        
        return memory_id
    
    def search_project_memories(self, 
                              query: str,
                              project_id: Optional[str] = None,
                              n_results: int = 5,
                              **kwargs) -> List[Dict[str, Any]]:
        """
        Search memories within a specific project or current project.
        
        Args:
            query: Search query
            project_id: Optional project ID (uses current if not specified)
            n_results: Number of results
            **kwargs: Additional search parameters
            
        Returns:
            List of memory results
        """
        # Save current project
        original_project = self.current_project.id
        
        # Switch to target project if specified
        if project_id and project_id != original_project:
            self.switch_project(project_id)
        
        try:
            # Perform search
            results = self.search_memory_enhanced(query, n_results=n_results, **kwargs)
            
            # Add project info to results
            for result in results:
                result['project_id'] = self.current_project.id
                result['project_name'] = self.current_project.name
            
            return results
            
        finally:
            # Restore original project if we switched
            if project_id and project_id != original_project:
                self.switch_project(original_project)
    
    def get_project_summary(self, project_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Get summary statistics for a project's memories.
        
        Args:
            project_id: Project ID (uses current if not specified)
            
        Returns:
            Summary statistics
        """
        if not project_id:
            project_id = self.current_project.id
        
        # Save current project
        original_project = self.current_project.id
        
        # Switch to target project
        if project_id != original_project:
            self.switch_project(project_id)
        
        try:
            # Get all memories
            all_memories = self.collection.get()
            
            # Analyze memories
            summary = {
                "project_id": project_id,
                "project_name": self.current_project.name,
                "total_memories": len(all_memories['ids']),
                "memory_types": {},
                "with_screen_captures": 0,
                "total_cost": 0.0,
                "date_range": {
                    "earliest": None,
                    "latest": None
                }
            }
            
            # Process each memory
            for i, metadata in enumerate(all_memories['metadatas']):
                # Count by type
                mem_type = metadata.get('memory_type', 'unknown')
                summary['memory_types'][mem_type] = summary['memory_types'].get(mem_type, 0) + 1
                
                # Count screen captures
                if metadata.get('context', {}).get('has_screen'):
                    summary['with_screen_captures'] += 1
                
                # Sum costs
                summary['total_cost'] += metadata.get('generation_cost', 0.0)
                
                # Track date range
                timestamp = metadata.get('timestamp')
                if timestamp:
                    if not summary['date_range']['earliest'] or timestamp < summary['date_range']['earliest']:
                        summary['date_range']['earliest'] = timestamp
                    if not summary['date_range']['latest'] or timestamp > summary['date_range']['latest']:
                        summary['date_range']['latest'] = timestamp
            
            return summary
            
        finally:
            # Restore original project
            if project_id != original_project:
                self.switch_project(original_project)
    
    def start_screen_monitoring(self, interval: int = 30):
        """
        Start monitoring screen for automatic context capture.
        
        Args:
            interval: Seconds between captures
        """
        def on_screen_change(image_bytes: bytes, description: str):
            """Callback for screen changes"""
            # Analyze screen content
            from .screen_capture import ScreenContextAnalyzer
            analyzer = ScreenContextAnalyzer(self.screen_capture)
            context = analyzer.analyze_screen_context(image_bytes)
            
            # If significant content detected, store as memory
            if context.get('suggestions'):
                self.remember_with_screen(
                    query=f"Screen monitoring: {description}",
                    response=f"Screen context captured: {', '.join(context['suggestions'])}",
                    screen_capture=image_bytes,
                    generation_cost=0.0,  # No LLM cost for monitoring
                    model_used="screen_monitor"
                )
        
        self.screen_capture.start_monitoring(
            callback=on_screen_change,
            interval=interval,
            detect_changes=True
        )
        
        logger.info(f"Started screen monitoring for project: {self.current_project.name}")
    
    def stop_screen_monitoring(self):
        """Stop screen monitoring"""
        self.screen_capture.stop_monitoring()
        logger.info("Stopped screen monitoring")