"""
Project-Aware Memory System for ADAM
Extends the base memory system with project isolation.

Consolidated from:
- src/adam/memory/project.py (v1 project-aware memory)
- src/adam_v2/services/memory_service.py (v2 project-scoped ChromaDB logic)
"""
import os
import hashlib
from typing import Optional, Dict, Any, List
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass
import logging

from .core import ADAMMemoryAdvanced, Memory, MemoryType

# Import project service (moved by Task 2)
try:
    from adam.services.project_service import ProjectManager, Project
except ImportError:
    # Stub if project_service not yet available
    ProjectManager = None
    Project = None

logger = logging.getLogger(__name__)


class ProjectAwareMemory:
    """
    Enhanced memory system that isolates memories by project.
    Each project gets its own ChromaDB collection for complete isolation.

    Consolidated from v1 ProjectAwareMemory and v2 ProjectMemoryService.
    Unlike the original v1 implementation, this does NOT subclass
    ADAMMemoryAdvanced to avoid requiring screen_capture and other
    deleted dependencies. Instead, it wraps the base memory system
    and delegates storage/retrieval through project-scoped collections.
    """

    def __init__(
        self,
        project_id: str,
        project_name: str = "default",
        persist_directory: str = "./data/adam_memory",
    ):
        self.project_id = project_id
        self.project_name = project_name
        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(parents=True, exist_ok=True)

        # Collection name scoped to project (from v2 pattern)
        self.collection_name = f"adam_project_{project_id[:8]}"

        # Initialize ChromaDB client
        try:
            import chromadb
            from chromadb.config import Settings

            self.chroma_client = chromadb.PersistentClient(
                path=str(self.persist_directory),
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True,
                ),
            )
            self._initialize_collection()
        except ImportError:
            logger.warning("ChromaDB not available; project memory will be non-functional")
            self.chroma_client = None
            self.collection = None

    def _initialize_collection(self):
        """Initialize the project's ChromaDB collection."""
        from .config import MemoryConfig

        try:
            self.collection = self.chroma_client.get_collection(name=self.collection_name)
            logger.info(f"Loaded existing project collection: {self.collection_name}")
        except Exception:
            memory_config = MemoryConfig()
            embedding_function = memory_config.get_embedding_function()
            self.collection = self.chroma_client.create_collection(
                name=self.collection_name,
                embedding_function=embedding_function,
                metadata={
                    "project_id": self.project_id,
                    "project_name": self.project_name,
                    "created_at": datetime.now().isoformat(),
                },
            )
            logger.info(f"Created new project collection: {self.collection_name}")

    # -- Storage --

    async def store_memory(
        self,
        content: str,
        memory_type: MemoryType,
        metadata: Optional[Dict[str, Any]] = None,
        conversation_id: Optional[str] = None,
        cost: Optional[float] = None,
    ) -> str:
        """Store a memory in the project's collection."""
        if not self.collection:
            return ""

        memory_id = self._generate_memory_id(content)
        memory_metadata = {
            "memory_type": memory_type.value if hasattr(memory_type, "value") else str(memory_type),
            "project_id": self.project_id,
            "project_name": self.project_name,
            "timestamp": datetime.now().isoformat(),
            "conversation_id": conversation_id or "",
            "generation_cost": cost or 0.0,
            "access_count": 0,
            "last_accessed": datetime.now().isoformat(),
        }
        if metadata:
            memory_metadata.update(metadata)

        self.collection.add(ids=[memory_id], documents=[content], metadatas=[memory_metadata])
        return memory_id

    # -- Search --

    async def search_memories(
        self,
        query: str,
        limit: int = 5,
        memory_types: Optional[List[MemoryType]] = None,
        min_relevance: float = 0.5,
        conversation_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Search memories in the project's collection."""
        if not self.collection:
            return []

        conditions = [{"project_id": {"$eq": self.project_id}}]
        if memory_types:
            type_values = [t.value if hasattr(t, "value") else str(t) for t in memory_types]
            conditions.append({"memory_type": {"$in": type_values}})
        if conversation_id:
            conditions.append({"conversation_id": {"$eq": conversation_id}})

        where_clause = {"$and": conditions} if len(conditions) > 1 else conditions[0]

        results = self.collection.query(
            query_texts=[query],
            n_results=limit * 2,
            where=where_clause,
        )

        search_results = []
        if results and results["ids"] and results["ids"][0]:
            for i, memory_id in enumerate(results["ids"][0]):
                distance = results["distances"][0][i] if results.get("distances") else 0
                relevance = 1 - (distance / 2)
                if relevance >= min_relevance:
                    metadata = results["metadatas"][0][i]
                    search_results.append(
                        {
                            "id": memory_id,
                            "content": results["documents"][0][i],
                            "metadata": metadata,
                            "relevance_score": relevance,
                            "memory_type": metadata.get("memory_type", "conversation"),
                            "timestamp": metadata.get("timestamp", datetime.now().isoformat()),
                        }
                    )
                    self._update_memory_access(memory_id)

        search_results.sort(key=lambda x: x["relevance_score"], reverse=True)
        return search_results[:limit]

    # -- Stats --

    async def get_memory_stats(self) -> Dict[str, Any]:
        """Get statistics about the project's memories."""
        if not self.collection:
            return {"total_memories": 0, "memory_types": {}, "total_cost": 0.0}

        all_memories = self.collection.get()
        if not all_memories or not all_memories["ids"]:
            return {"total_memories": 0, "memory_types": {}, "total_cost": 0.0}

        memory_types: Dict[str, int] = {}
        total_cost = 0.0
        for metadata in all_memories["metadatas"]:
            mt = metadata.get("memory_type", "unknown")
            memory_types[mt] = memory_types.get(mt, 0) + 1
            total_cost += metadata.get("generation_cost", 0.0)

        return {
            "total_memories": len(all_memories["ids"]),
            "memory_types": memory_types,
            "total_cost": round(total_cost, 4),
        }

    # -- File knowledge (from v2 memory_service) --

    async def add_file_knowledge(self, file_info: Dict[str, Any]) -> bool:
        """Add file knowledge to project memory."""
        if not self.collection:
            return False
        try:
            chunk_id = hashlib.sha256(
                f"{file_info['file_path']}:{file_info['chunk_name']}:{datetime.now().isoformat()}".encode()
            ).hexdigest()[:16]
            metadata = {
                "memory_type": "file_knowledge",
                "project_id": self.project_id,
                "file_path": file_info["file_path"],
                "chunk_name": file_info["chunk_name"],
                "file_type": file_info["file_type"],
                "timestamp": file_info["timestamp"],
                "importance": 0.7,
            }
            content = (
                f"File: {file_info['file_path']}\n"
                f"Component: {file_info['chunk_name']}\n"
                f"Type: {file_info['file_type']}\n\n"
                f"{file_info['content']}"
            )
            self.collection.add(ids=[chunk_id], documents=[content], metadatas=[metadata])
            return True
        except Exception as e:
            logger.error(f"Error adding file knowledge: {e}")
            return False

    async def search_file_knowledge(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Search file knowledge for a project."""
        if not self.collection:
            return []
        try:
            results = self.collection.query(
                query_texts=[query],
                where={
                    "$and": [
                        {"project_id": self.project_id},
                        {"memory_type": "file_knowledge"},
                    ]
                },
                n_results=limit,
            )
            if not results or not results["ids"]:
                return []
            memories = []
            for i in range(len(results["ids"][0])):
                memories.append(
                    {
                        "id": results["ids"][0][i],
                        "content": results["documents"][0][i],
                        "metadata": results["metadatas"][0][i],
                        "distance": results["distances"][0][i] if "distances" in results else 0,
                    }
                )
            return memories
        except Exception as e:
            logger.error(f"Error searching file knowledge: {e}")
            return []

    # -- Export / Import --

    async def export_memories(self) -> Dict[str, Any]:
        """Export all memories for backup or migration."""
        if not self.collection:
            return {"project_id": self.project_id, "memories": []}
        all_memories = self.collection.get()
        memories = []
        if all_memories and all_memories["ids"]:
            for i, mid in enumerate(all_memories["ids"]):
                memories.append(
                    {
                        "id": mid,
                        "content": all_memories["documents"][i],
                        "metadata": all_memories["metadatas"][i],
                    }
                )
        return {
            "project_id": self.project_id,
            "project_name": self.project_name,
            "export_date": datetime.now().isoformat(),
            "memory_count": len(memories),
            "memories": memories,
        }

    async def import_memories(self, export_data: Dict[str, Any]) -> int:
        """Import memories from an export."""
        imported = 0
        for memory in export_data.get("memories", []):
            try:
                self.collection.add(
                    ids=[memory["id"]],
                    documents=[memory["content"]],
                    metadatas=[memory["metadata"]],
                )
                imported += 1
            except Exception:
                continue
        return imported

    # -- Helpers --

    def _generate_memory_id(self, content: str) -> str:
        hash_input = f"{content}_{datetime.now().isoformat()}_{self.project_id}"
        return hashlib.sha256(hash_input.encode()).hexdigest()[:16]

    def _update_memory_access(self, memory_id: str):
        try:
            result = self.collection.get(ids=[memory_id])
            if result and result["metadatas"]:
                metadata = result["metadatas"][0]
                metadata["access_count"] = metadata.get("access_count", 0) + 1
                metadata["last_accessed"] = datetime.now().isoformat()
                self.collection.update(ids=[memory_id], metadatas=[metadata])
        except Exception:
            pass
