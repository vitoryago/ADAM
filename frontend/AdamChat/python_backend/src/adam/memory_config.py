"""
Memory Configuration for ADAM
Allows switching between different embedding models
"""
import os
from enum import Enum
from typing import Optional, Dict, Any
from dataclasses import dataclass

class EmbeddingProvider(Enum):
    SENTENCE_TRANSFORMER = "sentence_transformer"
    OPENAI = "openai"

@dataclass
class EmbeddingConfig:
    """Configuration for embedding models"""
    provider: EmbeddingProvider
    model_name: str
    dimension: int
    description: str
    
class MemoryConfig:
    """Central configuration for ADAM's memory system"""
    
    # Available embedding models
    EMBEDDING_MODELS = {
        # Sentence Transformer models (local, free)
        "all-MiniLM-L6-v2": EmbeddingConfig(
            provider=EmbeddingProvider.SENTENCE_TRANSFORMER,
            model_name="all-MiniLM-L6-v2",
            dimension=384,
            description="Fast, lightweight model. Good for general use."
        ),
        "all-mpnet-base-v2": EmbeddingConfig(
            provider=EmbeddingProvider.SENTENCE_TRANSFORMER,
            model_name="all-mpnet-base-v2",
            dimension=768,
            description="Higher quality embeddings, slower but more accurate."
        ),
        "all-MiniLM-L12-v2": EmbeddingConfig(
            provider=EmbeddingProvider.SENTENCE_TRANSFORMER,
            model_name="all-MiniLM-L12-v2",
            dimension=384,
            description="Balanced performance and quality."
        ),
        
        # OpenAI models (require API key, better quality)
        "text-embedding-3-small": EmbeddingConfig(
            provider=EmbeddingProvider.OPENAI,
            model_name="text-embedding-3-small",
            dimension=1536,
            description="OpenAI's small embedding model. Good balance of cost and quality."
        ),
        "text-embedding-3-large": EmbeddingConfig(
            provider=EmbeddingProvider.OPENAI,
            model_name="text-embedding-3-large",
            dimension=3072,
            description="OpenAI's large embedding model. Best quality but more expensive."
        ),
        "text-embedding-ada-002": EmbeddingConfig(
            provider=EmbeddingProvider.OPENAI,
            model_name="text-embedding-ada-002",
            dimension=1536,
            description="OpenAI's previous gen model. Still good quality."
        ),
    }
    
    def __init__(self):
        # Get preferred model from environment or use default
        self.embedding_model_name = os.getenv(
            "ADAM_EMBEDDING_MODEL", 
            "all-mpnet-base-v2"  # Better default than MiniLM
        )
        
        # Validate model exists
        if self.embedding_model_name not in self.EMBEDDING_MODELS:
            print(f"Warning: Unknown embedding model '{self.embedding_model_name}', using default")
            self.embedding_model_name = "all-mpnet-base-v2"
            
        self.embedding_config = self.EMBEDDING_MODELS[self.embedding_model_name]
        
    def get_embedding_function(self):
        """Get the appropriate embedding function based on provider"""
        if self.embedding_config.provider == EmbeddingProvider.SENTENCE_TRANSFORMER:
            # Use ChromaDB's sentence transformer embedding function
            from chromadb.utils import embedding_functions
            return embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name=self.embedding_config.model_name
            )
            
        elif self.embedding_config.provider == EmbeddingProvider.OPENAI:
            # Check if OpenAI key is available
            if not os.getenv("OPENAI_API_KEY"):
                print("Warning: OpenAI API key not found, falling back to local model")
                # Use ChromaDB's sentence transformer for fallback
                from chromadb.utils import embedding_functions
                fallback_config = self.EMBEDDING_MODELS["all-mpnet-base-v2"]
                return embedding_functions.SentenceTransformerEmbeddingFunction(
                    model_name=fallback_config.model_name
                )
            
            # Use ChromaDB's OpenAI embedding function
            from chromadb.utils import embedding_functions
            return embedding_functions.OpenAIEmbeddingFunction(
                api_key=os.getenv("OPENAI_API_KEY"),
                model_name=self.embedding_config.model_name
            )
    
    def get_info(self) -> Dict[str, Any]:
        """Get information about current configuration"""
        return {
            "model": self.embedding_model_name,
            "provider": self.embedding_config.provider.value,
            "dimension": self.embedding_config.dimension,
            "description": self.embedding_config.description
        }