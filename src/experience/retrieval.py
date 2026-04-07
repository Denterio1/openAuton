from __future__ import annotations
from typing import List, Optional
from experience.episodes import ExperienceEpisode, EpisodeStatus, TaskType


class EpisodeRetrieval:

    def __init__(self, history: List[ExperienceEpisode]) -> None:
        self.history = history

    def get_hints(self, task_type: TaskType, max_hints: int = 3) -> List[str]:
        hints = []
        for ep in reversed(self.history):
            if ep.task_type == task_type and ep.next_improvement:
                hints.append(ep.next_improvement)
            if len(hints) >= max_hints:
                break
        return hints

    def get_best(self, task_type: TaskType) -> Optional[ExperienceEpisode]:
        candidates = [
            ep for ep in self.history
            if ep.task_type == task_type
            and ep.evaluation
            and ep.evaluation.accuracy is not None
        ]
        if not candidates:
            return None
        return max(candidates, key=lambda e: e.evaluation.accuracy)

    def get_similar(self, task: str, task_type: TaskType, max_results: int = 3) -> List[ExperienceEpisode]:
        task_words = set(task.lower().split())
        scored = []
        for ep in self.history:
            ep_words = set(ep.task_description.lower().split())
            score = len(task_words & ep_words)
            if score > 0:
                scored.append((score, ep))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [ep for _, ep in scored[:max_results]]

    def success_rate(self, task_type: Optional[TaskType] = None) -> float:
        pool = [ep for ep in self.history if task_type is None or ep.task_type == task_type]
        if not pool:
            return 0.0
        return sum(1 for ep in pool if ep.status == EpisodeStatus.SUCCESS) / len(pool)

    def summary(self) -> str:
        return f"EpisodeRetrieval: {len(self.history)} episodes | success_rate={self.success_rate():.0%}"
    def semantic_search(self, query: str, top_k: int = 5) -> List[RetrievedEpisode]:
        """Search for episodes using semantic similarity (if embedding model is available)."""
        if self.embedding_model:
            query_embedding = self.embedding_model.encode(query)
            scores = []
            for ep_id, emb in self._embeddings_cache.items():
                sim = np.dot(query_embedding, emb) / (np.linalg.norm(query_embedding) * np.linalg.norm(emb))
                scores.append((ep_id, sim))
            scores.sort(key=lambda x: x[1], reverse=True)
            results = []
            for ep_id, sim in scores[:top_k]:
                ep = self._episode_cache.get(ep_id)
                if ep:
                    results.append(RetrievedEpisode(
                        episode=ep,
                        similarity_score=sim,
                        relevance_factors=["semantic_match"]
                    ))
            return results
        return []        