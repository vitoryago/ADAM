"""
Model-Native Web Search for ADAM
Uses the LLM's built-in web search capabilities instead of external APIs
"""

import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from datetime import datetime
import json

logger = logging.getLogger(__name__)

# Import the LLM client
try:
    from adam.llm.client import UnifiedLLMClient
    from adam.llm.config import LLMConfig
    CLIENT_AVAILABLE = True
except ImportError:
    CLIENT_AVAILABLE = False
    logger.warning("LLM client not available")


@dataclass
class SearchResult:
    """Result from model's web search"""
    query: str
    response: str
    sources: List[str] = None
    model_used: str = None
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


class ModelWebSearch:
    """
    Use the model's native web search capabilities
    This is more efficient than using external search APIs
    """
    
    def __init__(self):
        """Initialize with LLM client"""
        if not CLIENT_AVAILABLE:
            raise ImportError("LLM client required for model web search")
            
        self.client = UnifiedLLMClient()
        
    def search_with_grok(self, query: str, detailed: bool = False) -> SearchResult:
        """
        Use Grok's real-time web and X/Twitter access
        
        Grok has access to:
        - Real-time X/Twitter posts
        - Web search results
        - Current events and news
        """
        system_prompt = """You are a research assistant with access to real-time information.
        Provide accurate, up-to-date information from the web and social media.
        Always cite your sources when possible."""
        
        search_prompt = f"""Please search for and provide information about: {query}
        
        Include:
        1. Latest information available
        2. Key facts and findings
        3. Relevant sources or references
        {"4. Detailed analysis and context" if detailed else ""}
        
        Focus on accuracy and recency of information."""
        
        try:
            response = self.client.chat(
                prompt=search_prompt,
                model="grok-4",  # Grok-4 has web access
                system_prompt=system_prompt,
                temperature=0.3  # Lower temperature for factual search
            )
            
            return SearchResult(
                query=query,
                response=response.content,
                model_used="grok-4",
                sources=["Grok real-time web search", "X/Twitter data"]
            )
            
        except Exception as e:
            logger.error(f"Grok search failed: {e}")
            raise
    
    def search_with_openai(self, query: str, detailed: bool = False) -> SearchResult:
        """
        Use OpenAI's web browsing capability
        
        Note: Requires specific model versions with browsing enabled
        """
        system_prompt = """You are a research assistant. 
        Search the web for current information and provide accurate, well-sourced responses."""
        
        search_prompt = f"""Search the web for: {query}
        
        Provide:
        1. Current information (check dates)
        2. Multiple perspectives if relevant
        3. Source URLs when available
        {"4. In-depth analysis" if detailed else ""}"""
        
        try:
            # OpenAI models with browsing use special parameters
            response = self.client.chat(
                prompt=search_prompt,
                model="gpt-4-turbo-preview",  # Or gpt-4 with browsing
                system_prompt=system_prompt,
                temperature=0.3,
                # Note: OpenAI's browsing is enabled differently
                # This would need to be implemented in the client
                extra_params={"enable_browsing": True}
            )
            
            return SearchResult(
                query=query,
                response=response.content,
                model_used="gpt-4-turbo",
                sources=["OpenAI web browsing"]
            )
            
        except Exception as e:
            logger.error(f"OpenAI search failed: {e}")
            raise
    
    def search(self, query: str, preferred_model: str = "auto", 
              detailed: bool = False) -> SearchResult:
        """
        Search using the best available model with web access
        
        Args:
            query: Search query
            preferred_model: 'grok', 'openai', or 'auto'
            detailed: Whether to include detailed analysis
            
        Returns:
            SearchResult with the response
        """
        # Determine which model to use
        if preferred_model == "auto":
            # Check what's available and has web access
            available_models = self.client.config.get_available_models()
            
            if "grok-4" in available_models or "grok-4-reasoning" in available_models:
                preferred_model = "grok"
            elif "gpt-4" in available_models:
                preferred_model = "openai"
            else:
                raise ValueError("No models with web search capability available")
        
        # Execute search with selected model
        if preferred_model == "grok":
            return self.search_with_grok(query, detailed)
        elif preferred_model == "openai":
            return self.search_with_openai(query, detailed)
        else:
            raise ValueError(f"Unknown model preference: {preferred_model}")
    
    def search_and_summarize(self, query: str, focus_areas: List[str] = None) -> Dict[str, Any]:
        """
        Search and provide structured summary
        
        Args:
            query: Main search query
            focus_areas: Specific aspects to focus on
            
        Returns:
            Structured summary with key findings
        """
        # Build focused prompt
        focus_prompt = ""
        if focus_areas:
            focus_prompt = f"\n\nFocus especially on these aspects:\n" + \
                          "\n".join(f"- {area}" for area in focus_areas)
        
        # Get detailed search results
        result = self.search(query, detailed=True)
        
        # Parse the response to extract key information
        summary = {
            "query": query,
            "timestamp": result.timestamp.isoformat(),
            "model_used": result.model_used,
            "main_findings": result.response,
            "focus_areas": focus_areas,
            "sources": result.sources
        }
        
        # If we have focus areas, do a second pass to extract specific info
        if focus_areas:
            extraction_prompt = f"""Based on this information about {query}:

{result.response}

Please extract and organize information specifically about:
{chr(10).join(f"- {area}" for area in focus_areas)}

Format as a structured summary."""
            
            try:
                focused_response = self.client.chat(
                    prompt=extraction_prompt,
                    model="grok-3-mini-high",  # Use cheap model for extraction
                    temperature=0.2
                )
                summary["focused_insights"] = focused_response.content
            except:
                pass
        
        return summary
    
    def compare_sources(self, query: str) -> Dict[str, SearchResult]:
        """
        Search using multiple models and compare results
        Useful for getting diverse perspectives
        """
        results = {}
        
        # Try Grok
        try:
            results["grok"] = self.search_with_grok(query)
        except Exception as e:
            logger.warning(f"Grok search failed: {e}")
        
        # Try OpenAI
        try:
            results["openai"] = self.search_with_openai(query)
        except Exception as e:
            logger.warning(f"OpenAI search failed: {e}")
        
        return results


# Integration with existing ADAM system
class ADAMWebSearchIntegration:
    """
    Integrate model-native search with ADAM's memory system
    """
    
    def __init__(self, memory_system=None):
        self.search = ModelWebSearch()
        self.memory = memory_system
    
    def search_and_remember(self, query: str) -> SearchResult:
        """
        Search and store results in ADAM's memory
        """
        result = self.search.search(query)
        
        if self.memory:
            # Store the search results in memory for future reference
            self.memory.store(
                content=f"Web search for: {query}\n\nResults:\n{result.response}",
                metadata={
                    "type": "web_search",
                    "query": query,
                    "model": result.model_used,
                    "timestamp": result.timestamp.isoformat()
                }
            )
        
        return result
    
    def cached_search(self, query: str, max_age_hours: int = 24) -> SearchResult:
        """
        Check memory for recent searches before making new request
        """
        if self.memory:
            # Check if we have recent search results
            recent = self.memory.search(
                query=f"web search {query}",
                filter_metadata={"type": "web_search"},
                time_range_hours=max_age_hours
            )
            
            if recent:
                logger.info(f"Using cached search results for: {query}")
                return SearchResult(
                    query=query,
                    response=recent[0]["content"],
                    sources=["ADAM memory cache"],
                    model_used=recent[0].get("metadata", {}).get("model")
                )
        
        # No cache hit, perform new search
        return self.search_and_remember(query)


def quick_search(query: str) -> str:
    """
    Convenience function for quick web searches
    
    Example:
        result = quick_search("latest Python 3.13 features")
        print(result)
    """
    searcher = ModelWebSearch()
    result = searcher.search(query)
    return result.response


if __name__ == "__main__":
    # Demo the model-native search
    import sys
    
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
        print(f"\nSearching for: {query}\n")
        
        try:
            searcher = ModelWebSearch()
            result = searcher.search(query)
            
            print(f"Model used: {result.model_used}")
            print(f"Sources: {', '.join(result.sources)}")
            print(f"\nResults:\n{result.response}")
            
        except Exception as e:
            print(f"Search failed: {e}")
            print("\nMake sure you have GROK_API_KEY or OPENAI_API_KEY set")
    else:
        print("Usage: python model_web_search.py <search query>")
        print("\nExample: python model_web_search.py latest AI developments")