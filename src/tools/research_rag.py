"""
tools/research_rag.py
=====================
Dynamic RAG (Retrieval-Augmented Generation) tool for the agent.

Capabilities:
- Search arXiv, PubMed, or general web for latest research papers (SOTA).
- Cache results locally using file_ops.
- Build and query a vector store (FAISS or simple TF‑IDF) over cached papers.
- Extract architecture insights, training techniques, hyperparameters.
- Integrate with ml_engine to suggest novel model designs based on recent papers.

This tool gives the agent the ability to "keep up" with the latest ML research
and incorporate it into its planning and DNA evolution.
"""

from __future__ import annotations
import json
import re
import hashlib
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET

# Local tools (will be injected via registry)
from tools.file_ops import FileOps
from tools.python_executor import PythonExecutor


# ──────────────────────────────────────────────────────────
# Paper metadata
# ──────────────────────────────────────────────────────────

@dataclass
class ResearchPaper:
    """Metadata for a single research paper."""
    title: str
    authors: List[str]
    abstract: str
    url: str
    arxiv_id: Optional[str] = None
    published: Optional[str] = None
    categories: List[str] = field(default_factory=list)

    def summary(self) -> str:
        return f"{self.title} ({', '.join(self.authors[:3])}) – {self.abstract[:200]}..."

    def to_dict(self) -> Dict:
        return {
            "title": self.title,
            "authors": self.authors,
            "abstract": self.abstract,
            "url": self.url,
            "arxiv_id": self.arxiv_id,
            "published": self.published,
            "categories": self.categories,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'ResearchPaper':
        return cls(**data)


# ──────────────────────────────────────────────────────────
# Simple vector store (TF‑IDF + cosine similarity)
# ──────────────────────────────────────────────────────────

class SimpleVectorStore:
    """
    Lightweight vector store using TF‑IDF and cosine similarity.
    No external dependencies (FAISS optional but not required).
    """

    def __init__(self):
        self.papers: List[ResearchPaper] = []
        self.tfidf_vectors: List[Dict[str, float]] = []
        self.idf: Dict[str, float] = {}
        self.vocab: set = set()

    def add_papers(self, papers: List[ResearchPaper]):
        """Add papers and rebuild TF‑IDF index."""
        self.papers.extend(papers)
        self._build_index()

    def _build_index(self):
        # Build vocabulary from all abstracts + titles
        all_texts = [f"{p.title} {p.abstract}" for p in self.papers]
        self.vocab = set()
        for text in all_texts:
            words = re.findall(r'\w+', text.lower())
            self.vocab.update(words)

        # Compute term frequencies
        tf = []
        for text in all_texts:
            words = re.findall(r'\w+', text.lower())
            tf_dict = {}
            for w in words:
                tf_dict[w] = tf_dict.get(w, 0) + 1
            # normalize by max
            max_tf = max(tf_dict.values()) if tf_dict else 1
            tf.append({w: c / max_tf for w, c in tf_dict.items()})

        # Compute IDF
        doc_count = len(all_texts)
        self.idf = {}
        for word in self.vocab:
            docs_with_word = sum(1 for text in all_texts if word in text.lower())
            self.idf[word] = math.log((doc_count + 1) / (docs_with_word + 1)) + 1

        # Compute TF‑IDF vectors
        self.tfidf_vectors = []
        for tf_dict in tf:
            vec = {w: tf_val * self.idf.get(w, 1.0) for w, tf_val in tf_dict.items()}
            self.tfidf_vectors.append(vec)

    def search(self, query: str, top_k: int = 5) -> List[Tuple[ResearchPaper, float]]:
        """Return top_k papers with cosine similarity scores."""
        if not self.papers:
            return []
        # Compute query TF‑IDF
        query_words = re.findall(r'\w+', query.lower())
        q_tf = {}
        for w in query_words:
            q_tf[w] = q_tf.get(w, 0) + 1
        max_q = max(q_tf.values()) if q_tf else 1
        q_vec = {w: (c / max_q) * self.idf.get(w, 1.0) for w, c in q_tf.items()}

        # Compute cosine similarity with each paper
        similarities = []
        for i, paper_vec in enumerate(self.tfidf_vectors):
            # Dot product
            dot = sum(q_vec.get(w, 0) * paper_vec.get(w, 0) for w in set(q_vec) | set(paper_vec))
            norm_q = math.sqrt(sum(v*v for v in q_vec.values()))
            norm_p = math.sqrt(sum(v*v for v in paper_vec.values()))
            if norm_q == 0 or norm_p == 0:
                sim = 0.0
            else:
                sim = dot / (norm_q * norm_p)
            similarities.append((i, sim))
        similarities.sort(key=lambda x: x[1], reverse=True)
        return [(self.papers[i], sim) for i, sim in similarities[:top_k]]


# ──────────────────────────────────────────────────────────
# Main RAG Tool
# ──────────────────────────────────────────────────────────

class ResearchRAG:
    """
    Retrieves and indexes research papers from arXiv.
    Provides query interface for the agent to get SOTA insights.
    """

    ARXIV_API_URL = "http://export.arxiv.org/api/query"
    CACHE_DIR = Path("experiments/research_cache")

    def __init__(self, file_ops: FileOps, executor: Optional[PythonExecutor] = None):
        self.file_ops = file_ops
        self.executor = executor
        self.CACHE_DIR.mkdir(parents=True, exist_ok=True)
        self.vector_store = SimpleVectorStore()
        self._load_cache()

    def _load_cache(self):
        """Load previously cached papers from disk."""
        cache_file = self.CACHE_DIR / "papers_index.json"
        if cache_file.exists():
            data = self.file_ops.read_json(str(cache_file))
            if data:
                papers = [ResearchPaper.from_dict(p) for p in data.get("papers", [])]
                self.vector_store.add_papers(papers)

    def _save_cache(self):
        """Save paper index to disk."""
        papers_dict = [p.to_dict() for p in self.vector_store.papers]
        self.file_ops.write_json(str(self.CACHE_DIR / "papers_index.json"), {"papers": papers_dict})

    def search_arxiv(self, query: str, max_results: int = 10) -> List[ResearchPaper]:
        """
        Fetch papers from arXiv API based on query.
        """
        params = {
            "search_query": f"all:{query}",
            "start": 0,
            "max_results": max_results,
            "sortBy": "submittedDate",
            "sortOrder": "descending",
        }
        url = f"{self.ARXIV_API_URL}?{urllib.parse.urlencode(params)}"
        try:
            with urllib.request.urlopen(url, timeout=15) as response:
                data = response.read().decode('utf-8')
            return self._parse_arxiv_response(data)
        except Exception as e:
            print(f"arXiv search failed: {e}")
            return []

    def _parse_arxiv_response(self, xml_data: str) -> List[ResearchPaper]:
        """Parse arXiv API XML response."""
        papers = []
        # Simple regex extraction (more robust would use xml.etree)
        entries = re.findall(r'<entry>(.*?)</entry>', xml_data, re.DOTALL)
        for entry in entries:
            title_match = re.search(r'<title>(.*?)</title>', entry, re.DOTALL)
            title = title_match.group(1).strip() if title_match else "Unknown"
            # Remove newlines
            title = re.sub(r'\s+', ' ', title)

            authors = []
            for auth in re.findall(r'<name>(.*?)</name>', entry, re.DOTALL):
                authors.append(auth.strip())

            abstract_match = re.search(r'<summary>(.*?)</summary>', entry, re.DOTALL)
            abstract = abstract_match.group(1).strip() if abstract_match else ""
            abstract = re.sub(r'\s+', ' ', abstract)

            link_match = re.search(r'<id>(.*?)</id>', entry)
            url = link_match.group(1).strip() if link_match else ""

            arxiv_id = url.split('/')[-1] if url else None

            published_match = re.search(r'<published>(.*?)</published>', entry)
            published = published_match.group(1).strip() if published_match else None

            papers.append(ResearchPaper(
                title=title,
                authors=authors,
                abstract=abstract,
                url=url,
                arxiv_id=arxiv_id,
                published=published,
            ))
        return papers

    def retrieve(self, query: str, use_cache: bool = True, top_k: int = 5) -> List[Dict]:
        """
        Main retrieval method: search cache first, then arXiv if needed.
        Returns list of relevant paper summaries with similarity scores.
        """
        # Search local vector store
        local_results = self.vector_store.search(query, top_k=top_k)
        if local_results and use_cache:
            return [{"paper": p.to_dict(), "similarity": sim} for p, sim in local_results]

        # Fetch from arXiv
        new_papers = self.search_arxiv(query, max_results=top_k)
        if new_papers:
            self.vector_store.add_papers(new_papers)
            self._save_cache()
            # Re‑search to get scores
            fresh_results = self.vector_store.search(query, top_k=top_k)
            return [{"paper": p.to_dict(), "similarity": sim} for p, sim in fresh_results]
        return []

    def extract_insights(self, query: str) -> Dict[str, Any]:
        """
        Retrieve papers and extract actionable insights:
        - suggested architecture changes
        - novel training techniques
        - hyperparameter ranges
        """
        papers = self.retrieve(query, top_k=3)
        insights = {
            "query": query,
            "papers_found": len(papers),
            "architecture_hints": [],
            "training_hints": [],
            "hyperparameter_hints": [],
            "raw_papers": papers,
        }
        for item in papers:
            abstract = item["paper"]["abstract"].lower()
            # Simple keyword extraction
            if "transformer" in abstract or "attention" in abstract:
                insights["architecture_hints"].append("Consider transformer variants with attention")
            if "contrastive" in abstract:
                insights["training_hints"].append("Contrastive learning may improve representation")
            if "lora" in abstract or "qlora" in abstract:
                insights["training_hints"].append("LoRA/QLoRA for efficient fine‑tuning")
            if "learning rate" in abstract:
                # try to extract numeric value
                match = re.search(r'learning rate[^0-9]*([0-9.eE-]+)', abstract)
                if match:
                    insights["hyperparameter_hints"].append(f"learning_rate ~ {match.group(1)}")
        return insights

    def suggest_architecture(self, task_description: str) -> Dict[str, Any]:
        """
        High‑level suggestion: given a task, search for relevant papers and propose architecture.
        """
        results = self.retrieve(task_description, top_k=5)
        suggestions = {
            "task": task_description,
            "recommended_architecture": "unknown",
            "evidence": [],
            "papers_used": len(results),
        }
        # Simple heuristic: count architecture mentions
        arch_counts = {}
        for item in results:
            title = item["paper"]["title"].lower()
            if "transformer" in title:
                arch_counts["transformer"] = arch_counts.get("transformer", 0) + 1
            if "cnn" in title or "convolution" in title:
                arch_counts["cnn"] = arch_counts.get("cnn", 0) + 1
            if "mlp" in title or "resnet" in title:
                arch_counts["mlp"] = arch_counts.get("mlp", 0) + 1
        if arch_counts:
            best = max(arch_counts, key=arch_counts.get)
            suggestions["recommended_architecture"] = best
        return suggestions


# ──────────────────────────────────────────────────────────
# Tool Wrapper for Registry
# ──────────────────────────────────────────────────────────

def create_research_rag_tool(file_ops: FileOps, executor: PythonExecutor = None):
    """
    Factory to create a callable tool for the registry.
    """
    rag = ResearchRAG(file_ops, executor)

    def research_tool(query: str, action: str = "retrieve", top_k: int = 5) -> Dict:
        """
        Tool function for registry.
        action: "retrieve", "insights", "suggest"
        """
        if action == "retrieve":
            results = rag.retrieve(query, top_k=top_k)
            return {"status": "success", "results": results}
        elif action == "insights":
            insights = rag.extract_insights(query)
            return {"status": "success", "insights": insights}
        elif action == "suggest":
            suggestion = rag.suggest_architecture(query)
            return {"status": "success", "suggestion": suggestion}
        else:
            return {"status": "error", "message": f"Unknown action: {action}"}
    return research_tool