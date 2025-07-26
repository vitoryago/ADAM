"""
Project-based memory management for ADAM v2.0
Each project has its own isolated ChromaDB collection
"""

import chromadb
from chromadb.config import Settings
from typing import List, Dict, Any, Optional
from datetime import datetime
import logging

# Import from parent adam module when available
try:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from adam.memory import ADAMMemoryAdvanced
    from adam.advanced_rag import AdvancedRAGSystem
    ADAM_AVAILABLE = True
except ImportError:
    # For testing without full ADAM installation
    ADAM_AVAILABLE = False
    ADAMMemoryAdvanced = object

logger = logging.getLogger(__name__)

class ProjectMemoryManager:
    """Manages memories for a specific project"""
    
    def __init__(self, project_id: str):
        self.project_id = project_id
        self.collection_name = f"adam_project_{project_id}"
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(
            path="./adam_memory_projects",
            settings=Settings(anonymized_telemetry=False)
        )
        
        # Get or create collection
        try:
            self.collection = self.client.get_collection(self.collection_name)
        except:
            self.collection = None
    
    def initialize_collection(self):
        """Create a new collection for this project"""
        if self.collection is None:
            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata={
                    "project_id": self.project_id,
                    "created_at": datetime.now().isoformat()
                }
            )
            logger.info(f"Created memory collection for project {self.project_id}")
    
    def delete_collection(self):
        """Delete the collection when project is deleted"""
        try:
            self.client.delete_collection(self.collection_name)
            logger.info(f"Deleted memory collection for project {self.project_id}")
        except Exception as e:
            logger.error(f"Error deleting collection: {e}")
    
    def store_memory(
        self, 
        query: str, 
        response: str,
        conversation_id: str,
        memory_type: str = "conversation",
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Store a memory in the project's collection"""
        memory_id = f"mem_{self.project_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Prepare metadata
        full_metadata = {
            "project_id": self.project_id,
            "conversation_id": conversation_id,
            "memory_type": memory_type,
            "timestamp": datetime.now().isoformat(),
            "query": query,
            **(metadata or {})
        }
        
        # Store in ChromaDB
        self.collection.add(
            ids=[memory_id],
            documents=[f"Query: {query}\n\nResponse: {response}"],
            metadatas=[full_metadata]
        )
        
        return memory_id
    
    def search_memories(
        self,
        query: str,
        conversation_id: Optional[str] = None,
        n_results: int = 10
    ) -> List[Dict[str, Any]]:
        """Search memories within this project"""
        where_clause = {"project_id": self.project_id}
        if conversation_id:
            where_clause["conversation_id"] = conversation_id
        
        results = self.collection.query(
            query_texts=[query],
            where=where_clause,
            n_results=n_results
        )
        
        # Format results
        memories = []
        if results['documents'] and results['documents'][0]:
            for i, doc in enumerate(results['documents'][0]):
                memories.append({
                    'id': results['ids'][0][i],
                    'content': doc,
                    'metadata': results['metadatas'][0][i],
                    'distance': results['distances'][0][i]
                })
        
        return memories
    
    def get_project_stats(self) -> Dict[str, Any]:
        """Get memory statistics for this project"""
        total_memories = self.collection.count()
        
        # Get memory type distribution
        # Note: ChromaDB doesn't support aggregations, so we'd need to fetch all
        # For now, return basic stats
        return {
            "total_memories": total_memories,
            "collection_name": self.collection_name,
            "project_id": self.project_id
        }


if ADAM_AVAILABLE:
    class ProjectAwareMemorySystem(ADAMMemoryAdvanced):
        """
        Extended memory system that respects project boundaries
        Inherits from ADAM's advanced memory but adds project isolation
        """
        
        def __init__(self, project_id: str):
            self.project_id = project_id
            self.project_memory_manager = ProjectMemoryManager(project_id)
            
            # Initialize parent class with project-specific directory
            super().__init__(persist_directory=f"./adam_memory_projects/{project_id}")
            
            # Override the collection to use project-specific one
            self.collection = self.project_memory_manager.collection
else:
    # Stub for testing
    class ProjectAwareMemorySystem:
        def __init__(self, project_id: str):
            self.project_id = project_id
            self.project_memory_manager = ProjectMemoryManager(project_id)
    
    def remember_if_worthy(
        self,
        query: str,
        response: str,
        conversation_id: str,
        **kwargs
    ) -> Optional[str]:
        """Override to add conversation_id to metadata"""
        # Use parent's worthiness evaluation
        result = super().remember_if_worthy(query, response, **kwargs)
        
        if result and result.get('stored'):
            # Update metadata with conversation_id
            memory_id = result.get('memory_id')
            if memory_id:
                # Add conversation_id to the stored memory
                # This would need ChromaDB update functionality
                pass
        
        return result
    
    def recall_with_context(
        self,
        query: str,
        conversation_id: Optional[str] = None,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """Search memories with optional conversation filtering"""
        # Use project memory manager for isolated search
        return self.project_memory_manager.search_memories(
            query=query,
            conversation_id=conversation_id,
            n_results=kwargs.get('n_results', 10)
        )