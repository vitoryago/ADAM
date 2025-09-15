"""
Memory service for ADAM v2.0 - Project-based memory isolation
"""
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import asyncio
try:
    import chromadb
    from chromadb.config import Settings
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False
    chromadb = None
    Settings = None
import hashlib
import json
from dataclasses import dataclass
from enum import Enum

# Add parent directory to path for ADAM imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Try to import ADAM's memory components
try:
    from adam.memory.config import MemoryConfig
    from adam.memory.core import MemoryType
    ADAM_MEMORY_AVAILABLE = True
except ImportError:
    ADAM_MEMORY_AVAILABLE = False
    # Define minimal MemoryType for compatibility
    class MemoryType(Enum):
        ERROR_SOLUTION = "error_solution"
        CODE_PATTERN = "code_pattern"
        CONCEPT_EXPLANATION = "concept_explanation"
        SCREEN_ANALYSIS = "screen_analysis"
        EXPENSIVE_RESPONSE = "expensive_response"
        CONVERSATION = "conversation"


@dataclass
class MemorySearchResult:
    """Result from memory search"""
    id: str
    content: str
    metadata: Dict[str, Any]
    relevance_score: float
    memory_type: str
    timestamp: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "id": self.id,
            "content": self.content,
            "metadata": self.metadata,
            "relevance_score": self.relevance_score,
            "memory_type": self.memory_type,
            "timestamp": self.timestamp.isoformat()
        }


class ProjectMemoryService:
    """
    Manages project-specific memory isolation using ChromaDB.
    Each project gets its own collection for complete isolation.
    """
    
    def __init__(self, project_id: str, project_name: str):
        self.project_id = project_id
        self.project_name = project_name
        self.collection_name = f"adam_v2_project_{project_id[:8]}"
        
        # Initialize memory configuration
        if ADAM_MEMORY_AVAILABLE:
            self.memory_config = MemoryConfig()
        else:
            self.memory_config = None
        
        # Set up ChromaDB
        self.persist_directory = Path(os.getenv(
            "ADAM_V2_MEMORY_PATH",
            "./data/adam_v2_memory"
        ))
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        
        # Initialize ChromaDB client if available
        if CHROMADB_AVAILABLE:
            self.chroma_client = chromadb.PersistentClient(
                path=str(self.persist_directory),
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
            # Get or create collection
            self._initialize_collection()
        else:
            self.chroma_client = None
            self.collection = None
    
    def _initialize_collection(self):
        """Initialize the project's ChromaDB collection"""
        try:
            # Try to get existing collection
            self.collection = self.chroma_client.get_collection(
                name=self.collection_name
            )
        except Exception:
            # Create new collection with embedding function
            if self.memory_config and hasattr(self.memory_config, 'get_embedding_function'):
                embedding_function = self.memory_config.get_embedding_function()
            else:
                # Use default embedding function
                from chromadb.utils import embedding_functions
                embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
                    model_name="all-mpnet-base-v2"
                )
            
            self.collection = self.chroma_client.create_collection(
                name=self.collection_name,
                embedding_function=embedding_function,
                metadata={
                    "project_id": self.project_id,
                    "project_name": self.project_name,
                    "created_at": datetime.now().isoformat()
                }
            )
    
    async def store_memory(
        self,
        content: str,
        memory_type: MemoryType,
        metadata: Optional[Dict[str, Any]] = None,
        conversation_id: Optional[str] = None,
        cost: Optional[float] = None
    ) -> str:
        """Store a memory in the project's collection"""
        if not self.collection:
            return ""  # Memory not available
            
        # Generate unique ID
        memory_id = self._generate_memory_id(content)
        
        # Prepare metadata
        memory_metadata = {
            "memory_type": memory_type.value if hasattr(memory_type, 'value') else str(memory_type),
            "project_id": self.project_id,
            "project_name": self.project_name,
            "timestamp": datetime.now().isoformat(),
            "conversation_id": conversation_id,
            "generation_cost": cost or 0.0,
            "access_count": 0,
            "last_accessed": datetime.now().isoformat()
        }
        
        if metadata:
            memory_metadata.update(metadata)
        
        # Store in ChromaDB
        self.collection.add(
            ids=[memory_id],
            documents=[content],
            metadatas=[memory_metadata]
        )
        
        return memory_id
    
    async def search_memories(
        self,
        query: str,
        limit: int = 5,
        memory_types: Optional[List[MemoryType]] = None,
        min_relevance: float = 0.5,
        conversation_id: Optional[str] = None
    ) -> List[MemorySearchResult]:
        """Search memories in the project's collection"""
        if not self.collection:
            return []  # Memory not available
            
        # Build where clause for filtering
        # ChromaDB requires $and for multiple conditions
        conditions = [{"project_id": {"$eq": self.project_id}}]
        
        if memory_types:
            type_values = [
                t.value if hasattr(t, 'value') else str(t) 
                for t in memory_types
            ]
            conditions.append({"memory_type": {"$in": type_values}})
        
        if conversation_id:
            conditions.append({"conversation_id": {"$eq": conversation_id}})
        
        # If multiple conditions, use $and
        if len(conditions) > 1:
            where_clause = {"$and": conditions}
        else:
            where_clause = conditions[0] if conditions else None
        
        # Perform search
        results = self.collection.query(
            query_texts=[query],
            n_results=limit * 2,  # Get more to filter by relevance
            where=where_clause
        )
        
        # Parse results
        search_results = []
        if results and results['ids'] and results['ids'][0]:
            for i, memory_id in enumerate(results['ids'][0]):
                # Calculate relevance (ChromaDB distance to similarity)
                distance = results['distances'][0][i] if results['distances'] else 0
                relevance = 1 - (distance / 2)  # Convert to 0-1 scale
                
                if relevance >= min_relevance:
                    metadata = results['metadatas'][0][i]
                    
                    result = MemorySearchResult(
                        id=memory_id,
                        content=results['documents'][0][i],
                        metadata=metadata,
                        relevance_score=relevance,
                        memory_type=metadata.get('memory_type', 'conversation'),
                        timestamp=datetime.fromisoformat(
                            metadata.get('timestamp', datetime.now().isoformat())
                        )
                    )
                    search_results.append(result)
                    
                    # Update access count and last accessed
                    self._update_memory_access(memory_id)
        
        # Sort by relevance and limit
        search_results.sort(key=lambda x: x.relevance_score, reverse=True)
        return search_results[:limit]
    
    def _update_memory_access(self, memory_id: str):
        """Update memory access statistics"""
        try:
            # Get current metadata
            result = self.collection.get(ids=[memory_id])
            if result and result['metadatas']:
                metadata = result['metadatas'][0]
                metadata['access_count'] = metadata.get('access_count', 0) + 1
                metadata['last_accessed'] = datetime.now().isoformat()
                
                # Update metadata
                self.collection.update(
                    ids=[memory_id],
                    metadatas=[metadata]
                )
        except Exception:
            # Non-critical error, don't fail the search
            pass
    
    async def get_memory_stats(self) -> Dict[str, Any]:
        """Get statistics about the project's memories"""
        if not self.collection:
            return {
                "total_memories": 0,
                "memory_types": {},
                "total_cost": 0.0,
                "avg_access_count": 0.0,
                "oldest_memory": None,
                "newest_memory": None
            }
            
        # Get all memories for stats
        all_memories = self.collection.get()
        
        if not all_memories or not all_memories['ids']:
            return {
                "total_memories": 0,
                "memory_types": {},
                "total_cost": 0.0,
                "avg_access_count": 0.0,
                "oldest_memory": None,
                "newest_memory": None
            }
        
        # Calculate statistics
        memory_types = {}
        total_cost = 0.0
        total_access = 0
        oldest = None
        newest = None
        
        for metadata in all_memories['metadatas']:
            # Count by type
            mem_type = metadata.get('memory_type', 'unknown')
            memory_types[mem_type] = memory_types.get(mem_type, 0) + 1
            
            # Sum costs
            total_cost += metadata.get('generation_cost', 0.0)
            
            # Sum access counts
            total_access += metadata.get('access_count', 0)
            
            # Track timestamps
            timestamp = metadata.get('timestamp')
            if timestamp:
                ts = datetime.fromisoformat(timestamp)
                if not oldest or ts < oldest:
                    oldest = ts
                if not newest or ts > newest:
                    newest = ts
        
        total_memories = len(all_memories['ids'])
        
        return {
            "total_memories": total_memories,
            "memory_types": memory_types,
            "total_cost": round(total_cost, 4),
            "avg_access_count": round(total_access / total_memories, 2) if total_memories > 0 else 0,
            "oldest_memory": oldest.isoformat() if oldest else None,
            "newest_memory": newest.isoformat() if newest else None
        }
    
    async def clear_memories(self) -> int:
        """Clear all memories for the project"""
        # Get count before deletion
        all_memories = self.collection.get()
        count = len(all_memories['ids']) if all_memories and all_memories['ids'] else 0
        
        # Delete the collection
        self.chroma_client.delete_collection(name=self.collection_name)
        
        # Recreate empty collection
        self._initialize_collection()
        
        return count
    
    def _generate_memory_id(self, content: str) -> str:
        """Generate a unique ID for a memory"""
        # Create hash from content and timestamp
        hash_input = f"{content}_{datetime.now().isoformat()}_{self.project_id}"
        return hashlib.sha256(hash_input.encode()).hexdigest()[:16]
    
    async def export_memories(self) -> Dict[str, Any]:
        """Export all memories for backup or migration"""
        all_memories = self.collection.get()
        
        if not all_memories or not all_memories['ids']:
            return {
                "project_id": self.project_id,
                "project_name": self.project_name,
                "export_date": datetime.now().isoformat(),
                "memories": []
            }
        
        memories = []
        for i, memory_id in enumerate(all_memories['ids']):
            memories.append({
                "id": memory_id,
                "content": all_memories['documents'][i],
                "metadata": all_memories['metadatas'][i]
            })
        
        return {
            "project_id": self.project_id,
            "project_name": self.project_name,
            "export_date": datetime.now().isoformat(),
            "memory_count": len(memories),
            "memories": memories
        }
    
    async def import_memories(self, export_data: Dict[str, Any]) -> int:
        """Import memories from an export"""
        imported = 0
        
        for memory in export_data.get('memories', []):
            try:
                self.collection.add(
                    ids=[memory['id']],
                    documents=[memory['content']],
                    metadatas=[memory['metadata']]
                )
                imported += 1
            except Exception:
                # Skip duplicates or errors
                continue
        
        return imported
    
    async def add_file_knowledge(self, project_id: str, file_info: Dict[str, Any]) -> bool:
        """Add file knowledge to memory"""
        if not CHROMADB_AVAILABLE or not self.collection:
            return False
        
        try:
            # Generate unique ID for this chunk
            chunk_id = hashlib.sha256(
                f"{file_info['file_path']}:{file_info['chunk_name']}:{datetime.now().isoformat()}".encode()
            ).hexdigest()[:16]
            
            metadata = {
                "memory_type": "file_knowledge",
                "project_id": project_id,
                "file_path": file_info['file_path'],
                "chunk_name": file_info['chunk_name'],
                "file_type": file_info['file_type'],
                "timestamp": file_info['timestamp'],
                "importance": 0.7  # Medium importance for file knowledge
            }
            
            # Create searchable content
            content = f"File: {file_info['file_path']}\n"
            content += f"Component: {file_info['chunk_name']}\n"
            content += f"Type: {file_info['file_type']}\n\n"
            content += file_info['content']
            
            # Add to collection
            self.collection.add(
                ids=[chunk_id],
                documents=[content],
                metadatas=[metadata]
            )
            
            return True
            
        except Exception as e:
            print(f"Error adding file knowledge: {e}")
            return False
    
    async def remove_file_knowledge(self, project_id: str, file_path: str) -> bool:
        """Remove file knowledge from memory"""
        if not CHROMADB_AVAILABLE or not self.collection:
            return False
        
        try:
            # Query for all chunks from this file
            results = self.collection.get(
                where={
                    "$and": [
                        {"project_id": project_id},
                        {"file_path": file_path}
                    ]
                }
            )
            
            if results and results['ids']:
                # Delete all chunks from this file
                self.collection.delete(ids=results['ids'])
                return True
            
            return False
            
        except Exception as e:
            print(f"Error removing file knowledge: {e}")
            return False
    
    async def search_file_knowledge(self, query: str, project_id: str, limit: int = 5) -> List[Dict]:
        """Search file knowledge for a project"""
        if not CHROMADB_AVAILABLE or not self.collection:
            return []
        
        try:
            results = self.collection.query(
                query_texts=[query],
                where={
                    "$and": [
                        {"project_id": project_id},
                        {"memory_type": "file_knowledge"}
                    ]
                },
                n_results=limit
            )
            
            if not results or not results['ids']:
                return []
            
            memories = []
            for i in range(len(results['ids'][0])):
                memories.append({
                    "id": results['ids'][0][i],
                    "content": results['documents'][0][i],
                    "metadata": results['metadatas'][0][i],
                    "distance": results['distances'][0][i] if 'distances' in results else 0
                })
            
            return memories
            
        except Exception as e:
            print(f"Error searching file knowledge: {e}")
            return []