"""
tools/web_search.py
===================
Web search tool for the agent. Uses DuckDuckGo (no API key required) or falls back to a custom endpoint.
Integrates with ToolRegistry and records search queries in episodes.
"""

from __future__ import annotations
import json
import asyncio
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import logging

try:
    from duckduckgo_search import DDGS
    HAS_DDG = True
except ImportError:
    HAS_DDG = False

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    title: str
    url: str
    snippet: str
    source: str = "duckduckgo"


class WebSearch:
    """
    Performs web searches using DuckDuckGo (free, no API key).
    Returns top results as structured data.
    """

    def __init__(self, max_results: int = 5, timeout: int = 10):
        self.max_results = max_results
        self.timeout = timeout

    def search(self, query: str) -> List[SearchResult]:
        """
        Synchronous search – returns list of SearchResult.
        """
        if not HAS_DDG:
            logger.warning("duckduckgo-search not installed. Install with: pip install duckduckgo-search")
            return []

        try:
            with DDGS(timeout=self.timeout) as ddgs:
                results = []
                for r in ddgs.text(query, max_results=self.max_results):
                    results.append(SearchResult(
                        title=r.get('title', ''),
                        url=r.get('href', ''),
                        snippet=r.get('body', ''),
                        source='duckduckgo'
                    ))
                return results
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []

    async def search_async(self, query: str) -> List[SearchResult]:
        """
        Async wrapper – runs sync search in a thread.
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.search, query)

    def format_results(self, results: List[SearchResult]) -> str:
        """Format search results for LLM consumption."""
        if not results:
            return "No results found."
        lines = []
        for i, r in enumerate(results, 1):
            lines.append(f"{i}. **{r.title}**\n   URL: {r.url}\n   {r.snippet[:200]}...")
        return "\n\n".join(lines)


# Tool wrapper for registry
def create_web_search_tool():
    searcher = WebSearch(max_results=5, timeout=10)

    def web_search(query: str, max_results: int = 5) -> Dict[str, Any]:
        """
        Tool function: search the web for given query.
        """
        searcher.max_results = max_results
        results = searcher.search(query)
        return {
            "status": "success",
            "query": query,
            "num_results": len(results),
            "results": [{"title": r.title, "url": r.url, "snippet": r.snippet} for r in results],
            "formatted": searcher.format_results(results),
        }
    return web_search