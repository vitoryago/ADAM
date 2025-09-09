"""
Web Search Tool for ADAM
Provides internet research capabilities with multiple search providers
"""

import os
import json
import asyncio
import logging
from typing import List, Dict, Optional, Any
from datetime import datetime
from dataclasses import dataclass
import hashlib

logger = logging.getLogger(__name__)

# Try to import search providers
PROVIDERS_AVAILABLE = {}

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False
    logger.warning("requests not installed. Run: pip install requests")

try:
    from tavily import TavilyClient
    PROVIDERS_AVAILABLE['tavily'] = True
except ImportError:
    PROVIDERS_AVAILABLE['tavily'] = False
    logger.info("Tavily not available. Run: pip install tavily-python")

try:
    from serpapi import GoogleSearch
    PROVIDERS_AVAILABLE['serpapi'] = True
except ImportError:
    PROVIDERS_AVAILABLE['serpapi'] = False
    logger.info("SerpAPI not available. Run: pip install google-search-results")


@dataclass
class SearchResult:
    """Structured search result"""
    title: str
    url: str
    snippet: str
    content: Optional[str] = None
    relevance_score: float = 0.0
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()
            
    def to_dict(self):
        return {
            'title': self.title,
            'url': self.url,
            'snippet': self.snippet,
            'content': self.content,
            'relevance_score': self.relevance_score,
            'timestamp': self.timestamp.isoformat()
        }


class WebSearchTool:
    """
    Unified web search interface supporting multiple providers
    """
    
    def __init__(self, provider: str = 'auto', cache_results: bool = True):
        """
        Initialize web search tool
        
        Args:
            provider: 'tavily', 'serpapi', 'duckduckgo', or 'auto'
            cache_results: Whether to cache search results
        """
        self.provider = self._select_provider(provider)
        self.cache_results = cache_results
        self.cache = {} if cache_results else None
        self.clients = {}
        self._initialize_clients()
        
    def _select_provider(self, provider: str) -> str:
        """Select the best available provider"""
        if provider == 'auto':
            # Priority order
            if PROVIDERS_AVAILABLE.get('tavily'):
                return 'tavily'
            elif PROVIDERS_AVAILABLE.get('serpapi'):
                return 'serpapi'
            else:
                return 'duckduckgo'  # Fallback - no API key needed
        return provider
        
    def _initialize_clients(self):
        """Initialize search API clients"""
        if self.provider == 'tavily' and PROVIDERS_AVAILABLE.get('tavily'):
            api_key = os.getenv('TAVILY_API_KEY')
            if api_key:
                self.clients['tavily'] = TavilyClient(api_key=api_key)
            else:
                logger.warning("TAVILY_API_KEY not found in environment")
                
        elif self.provider == 'serpapi' and PROVIDERS_AVAILABLE.get('serpapi'):
            api_key = os.getenv('SERPAPI_API_KEY')
            if api_key:
                self.clients['serpapi'] = api_key
            else:
                logger.warning("SERPAPI_API_KEY not found in environment")
    
    def _get_cache_key(self, query: str, options: Dict) -> str:
        """Generate cache key for search query"""
        cache_data = f"{query}:{json.dumps(options, sort_keys=True)}"
        return hashlib.md5(cache_data.encode()).hexdigest()
    
    async def search_async(self, query: str, max_results: int = 5, 
                           include_content: bool = False) -> List[SearchResult]:
        """
        Async search with selected provider
        """
        return await asyncio.to_thread(
            self.search, query, max_results, include_content
        )
    
    def search(self, query: str, max_results: int = 5, 
              include_content: bool = False) -> List[SearchResult]:
        """
        Perform web search with selected provider
        
        Args:
            query: Search query
            max_results: Maximum number of results
            include_content: Whether to fetch full page content
            
        Returns:
            List of SearchResult objects
        """
        # Check cache
        if self.cache_results:
            cache_key = self._get_cache_key(query, {'max': max_results})
            if cache_key in self.cache:
                logger.info(f"Returning cached results for: {query}")
                return self.cache[cache_key]
        
        results = []
        
        try:
            if self.provider == 'tavily':
                results = self._search_tavily(query, max_results, include_content)
            elif self.provider == 'serpapi':
                results = self._search_serpapi(query, max_results)
            else:
                results = self._search_duckduckgo(query, max_results)
                
        except Exception as e:
            logger.error(f"Search failed with {self.provider}: {e}")
            # Fallback to DuckDuckGo
            if self.provider != 'duckduckgo':
                logger.info("Falling back to DuckDuckGo search")
                results = self._search_duckduckgo(query, max_results)
        
        # Cache results
        if self.cache_results and results:
            self.cache[cache_key] = results
            
        return results
    
    def _search_tavily(self, query: str, max_results: int, 
                      include_content: bool) -> List[SearchResult]:
        """Search using Tavily API"""
        if 'tavily' not in self.clients:
            raise ValueError("Tavily client not initialized")
            
        response = self.clients['tavily'].search(
            query=query,
            max_results=max_results,
            include_answer=True,
            include_raw_content=include_content
        )
        
        results = []
        for item in response.get('results', []):
            results.append(SearchResult(
                title=item.get('title', ''),
                url=item.get('url', ''),
                snippet=item.get('content', '')[:500],
                content=item.get('raw_content') if include_content else None,
                relevance_score=item.get('score', 0.0)
            ))
            
        return results
    
    def _search_serpapi(self, query: str, max_results: int) -> List[SearchResult]:
        """Search using SerpAPI (Google)"""
        if 'serpapi' not in self.clients:
            raise ValueError("SerpAPI key not initialized")
            
        search = GoogleSearch({
            'q': query,
            'num': max_results,
            'api_key': self.clients['serpapi']
        })
        
        response = search.get_dict()
        results = []
        
        for item in response.get('organic_results', []):
            results.append(SearchResult(
                title=item.get('title', ''),
                url=item.get('link', ''),
                snippet=item.get('snippet', ''),
                relevance_score=item.get('position', 0) / 100.0
            ))
            
        return results
    
    def _search_duckduckgo(self, query: str, max_results: int) -> List[SearchResult]:
        """
        Search using DuckDuckGo (no API key required)
        Simple HTTP-based search as fallback
        """
        if not REQUESTS_AVAILABLE:
            raise ImportError("requests library required for DuckDuckGo search")
            
        import requests
        from urllib.parse import quote
        
        # DuckDuckGo instant answer API (limited but free)
        url = f"https://api.duckduckgo.com/?q={quote(query)}&format=json&no_html=1"
        
        try:
            response = requests.get(url, timeout=10)
            data = response.json()
            
            results = []
            
            # Get abstract if available
            if data.get('Abstract'):
                results.append(SearchResult(
                    title=data.get('Heading', query),
                    url=data.get('AbstractURL', ''),
                    snippet=data.get('Abstract', ''),
                    relevance_score=1.0
                ))
            
            # Get related topics
            for topic in data.get('RelatedTopics', [])[:max_results-1]:
                if isinstance(topic, dict) and 'Text' in topic:
                    results.append(SearchResult(
                        title=topic.get('Text', '').split(' - ')[0][:100],
                        url=topic.get('FirstURL', ''),
                        snippet=topic.get('Text', ''),
                        relevance_score=0.8
                    ))
            
            # If no results, do a basic web scrape (less reliable)
            if not results:
                logger.info(f"No DuckDuckGo instant answers for: {query}")
                
            return results[:max_results]
            
        except Exception as e:
            logger.error(f"DuckDuckGo search failed: {e}")
            return []
    
    def search_and_summarize(self, query: str, max_results: int = 3) -> Dict[str, Any]:
        """
        Search and provide a structured summary
        
        Returns:
            Dictionary with query, results, and summary
        """
        results = self.search(query, max_results=max_results, include_content=True)
        
        summary = {
            'query': query,
            'timestamp': datetime.now().isoformat(),
            'num_results': len(results),
            'sources': [],
            'key_findings': []
        }
        
        for result in results:
            summary['sources'].append({
                'title': result.title,
                'url': result.url,
                'relevance': result.relevance_score
            })
            
            # Extract key points from snippet
            if result.snippet:
                # Simple extraction - could be enhanced with NLP
                sentences = result.snippet.split('. ')[:2]
                summary['key_findings'].extend(sentences)
        
        return summary
    
    def clear_cache(self):
        """Clear the search cache"""
        if self.cache:
            self.cache.clear()
            logger.info("Search cache cleared")


# Convenience function for quick searches
def quick_search(query: str, provider: str = 'auto') -> List[Dict]:
    """
    Quick search function for simple use cases
    
    Example:
        results = quick_search("latest Python features")
        for r in results:
            print(f"{r['title']}: {r['snippet']}")
    """
    tool = WebSearchTool(provider=provider)
    results = tool.search(query, max_results=5)
    return [r.to_dict() for r in results]


if __name__ == "__main__":
    # Test the search tool
    import sys
    
    if len(sys.argv) > 1:
        query = ' '.join(sys.argv[1:])
        print(f"\nSearching for: {query}\n")
        
        tool = WebSearchTool()
        results = tool.search(query, max_results=3)
        
        for i, result in enumerate(results, 1):
            print(f"{i}. {result.title}")
            print(f"   URL: {result.url}")
            print(f"   {result.snippet[:200]}...")
            print()
    else:
        print("Usage: python web_search.py <search query>")