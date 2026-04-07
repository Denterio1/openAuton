from __future__ import annotations

import json
import uuid
import time
import shutil
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional


# ─────────────────────────────────────────────
# Enums
# ─────────────────────────────────────────────

class EpisodeStatus(str, Enum):
    RUNNING = "running"
    SUCCESS = "success"
    FAILED  = "failed"
    PARTIAL = "partial"


class TaskType(str, Enum):
    TRAIN_FROM_SCRATCH  = "train_from_scratch"
    FINE_TUNE           = "fine_tune"
    ARCHITECTURE_DESIGN = "architecture_design"
    EVALUATION          = "evaluation"
    SELF_IMPROVEMENT    = "self_improvement"
    MULTI_AGENT_DESIGN  = "multi_agent_design"
    TOOL_COMPOSITION    = "tool_composition"
    CODE_GENERATION     = "code_generation"
    UNKNOWN             = "unknown"


class FailureMode(str, Enum):
    NONE              = "none"
    HALLUCINATION     = "hallucination"
    TOOL_ERROR        = "tool_error"
    PLANNING_ERROR    = "planning_error"
    DATA_QUALITY      = "data_quality"
    SAFETY_VIOLATION  = "safety_violation"
    TIMEOUT           = "timeout"
    LLM_ERROR         = "llm_error"
    UNKNOWN           = "unknown"


# ─────────────────────────────────────────────
# Sub-dataclasses
# ─────────────────────────────────────────────

@dataclass
class ModelSnapshot:
    architecture_name: str
    param_count: int
    hyperparams: Dict[str, Any]
    data_description: str
    objective: str
    notes: str = ""


@dataclass
class EvaluationResult:
    accuracy: Optional[float]           = None
    reasoning_quality: Optional[float]  = None
    hallucination_rate: Optional[float] = None
    safety_score: Optional[float]       = None
    plan_efficiency: Optional[float]    = None
    tool_usage_correct: bool            = True
    learned_from_history: bool          = False
    duration_seconds: float             = 0.0
    token_cost_estimate: int            = 0
    notes: str                          = ""


@dataclass
class ToolCallRecord:
    tool_name: str
    input_summary: str
    output_summary: str
    success: bool
    duration_seconds: float = 0.0
    error_message: str      = ""


@dataclass
class LLMCallRecord:
    purpose: str
    prompt_summary: str
    response_summary: str
    model: str
    provider: str
    input_tokens: int       = 0
    output_tokens: int      = 0
    cost_usd: float         = 0.0
    duration_seconds: float = 0.0
    success: bool           = True
    error: str              = ""

    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


@dataclass
class CausalStep:
    decision: str           # what the agent decided
    reason: str             # why it decided this
    evidence: str           # what data/history supported this decision
    confidence: float = 0.5 # 0.0–1.0


# ─────────────────────────────────────────────
# Pricing
# ─────────────────────────────────────────────

_PRICING: Dict[str, Dict[str, float]] = {
    "groq": {
        "llama-3.3-70b-versatile": 0.59,
        "llama-3.1-8b-instant":    0.05,
        "mixtral-8x7b-32768":      0.24,
        "gemma2-9b-it":            0.20,
        "default":                 0.59,
    },
    "openai": {
        "gpt-4o":       5.00,
        "gpt-4o-mini":  0.15,
        "default":      5.00,
    },
    "anthropic": {
        "claude-3-5-sonnet-20241022": 3.00,
        "claude-3-5-haiku-20241022":  0.80,
        "default":                    3.00,
    },
    "ollama": {"default": 0.0},
}


def estimate_cost_usd(provider: str, model: str, total_tokens: int) -> float:
    p = _PRICING.get(provider.lower(), {})
    rate = p.get(model, p.get("default", 0.0))
    return round((total_tokens / 1_000_000) * rate, 8)


# ─────────────────────────────────────────────
# Main ExperienceEpisode
# ─────────────────────────────────────────────

@dataclass
class ExperienceEpisode:
    """
    Atomic memory unit. One episode = one full task attempt.
    AGI additions vs AutoGPT/Voyager/OpenClaw:
      - causal_trace: WHY each decision was made
      - counterfactual_options: alternatives the agent rejected
      - uncertainty_score: agent confidence 0–1
      - failure_mode: typed enum, not a string
      - parent_episode_id: builds an evolution tree across runs
    """

    # Identity
    episode_id: str              = field(default_factory=lambda: str(uuid.uuid4()))
    task_description: str        = ""
    task_type: TaskType          = TaskType.UNKNOWN

    # Timing
    started_at: float            = field(default_factory=time.time)
    finished_at: Optional[float] = None
    status: EpisodeStatus        = EpisodeStatus.RUNNING

    # Execution trace
    plan_steps: List[str]                = field(default_factory=list)
    tool_calls: List[ToolCallRecord]     = field(default_factory=list)
    llm_calls:  List[LLMCallRecord]      = field(default_factory=list)

    # Model state
    model_before: Optional[ModelSnapshot] = None
    model_after:  Optional[ModelSnapshot] = None

    # Evaluation
    evaluation: Optional[EvaluationResult] = None

    # Human feedback
    human_feedback: str         = ""
    human_rating: Optional[int] = None

    # Agent reflection
    agent_reflection: str = ""
    next_improvement: str = ""

    # Genome
    gene_hints: List[str] = field(default_factory=list)

    # Cost
    cost_usd: float = 0.0

    # Tags
    tags: List[str] = field(default_factory=list)


    # Records the causal chain: why each decision was made
    causal_trace: List[CausalStep] = field(default_factory=list)

    # Alternatives the agent considered but rejected — enables counterfactual learning
    counterfactual_options: List[str] = field(default_factory=list)

    # Agent's confidence in its own plan (0=no idea, 1=certain)
    uncertainty_score: float = 0.5

    # Typed failure classification — downstream genome/retro.py uses this
    failure_mode: FailureMode = FailureMode.NONE

    # Links this episode to a previous one — builds an evolution tree
    parent_episode_id: Optional[str] = None

    # ─────────────────────────────────────────
    # Factory
    # ─────────────────────────────────────────

    @classmethod
    def start(
        cls,
        task: str,
        task_type: TaskType       = TaskType.UNKNOWN,
        tags: Optional[List[str]] = None,
        parent_id: Optional[str]  = None,
        uncertainty: float        = 0.5,
    ) -> "ExperienceEpisode":
        return cls(
            task_description=task,
            task_type=task_type,
            started_at=time.time(),
            status=EpisodeStatus.RUNNING,
            tags=tags or [],
            parent_episode_id=parent_id,
            uncertainty_score=uncertainty,
        )

    # ─────────────────────────────────────────
    # Mutation helpers
    # ─────────────────────────────────────────

    def add_plan_step(self, step: str) -> None:
        self.plan_steps.append(step)

    def record_tool(
        self,
        tool_name: str,
        input_summary: str,
        output_summary: str,
        success: bool,
        duration: float = 0.0,
        error: str      = "",
    ) -> None:
        self.tool_calls.append(ToolCallRecord(
            tool_name=tool_name,
            input_summary=input_summary[:300],
            output_summary=output_summary[:300],
            success=success,
            duration_seconds=duration,
            error_message=error,
        ))

    def record_llm_call(
        self,
        purpose: str,
        prompt_summary: str,
        response_summary: str,
        model: str,
        provider: str,
        input_tokens: int       = 0,
        output_tokens: int      = 0,
        duration_seconds: float = 0.0,
        success: bool           = True,
        error: str              = "",
    ) -> None:
        total  = input_tokens + output_tokens
        cost   = estimate_cost_usd(provider, model, total)
        self.llm_calls.append(LLMCallRecord(
            purpose=purpose,
            prompt_summary=prompt_summary[:200],
            response_summary=response_summary[:200],
            model=model,
            provider=provider,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost,
            duration_seconds=duration_seconds,
            success=success,
            error=error,
        ))
        self.cost_usd = round(self.cost_usd + cost, 8)

    def add_gene_hint(self, hint: str) -> None:
        if hint and hint not in self.gene_hints:
            self.gene_hints.append(hint)

    def record_causal_step(
        self,
        decision: str,
        reason: str,
        evidence: str = "",
        confidence: float = 0.5,
    ) -> None:
        self.causal_trace.append(CausalStep(decision, reason, evidence, confidence))

    def add_counterfactual(self, option: str) -> None:
        if option and option not in self.counterfactual_options:
            self.counterfactual_options.append(option)

    def set_model_snapshot(self, snapshot: ModelSnapshot, phase: str = "after") -> None:
        if phase == "before":
            self.model_before = snapshot
        else:
            self.model_after  = snapshot

    def finish(
        self,
        status: EpisodeStatus,
        evaluation: Optional[EvaluationResult] = None,
        reflection: str       = "",
        next_improvement: str = "",
        failure_mode: FailureMode = FailureMode.NONE,
    ) -> None:
        self.finished_at      = time.time()
        self.status           = status
        self.evaluation       = evaluation
        self.agent_reflection = reflection
        self.next_improvement = next_improvement
        self.failure_mode     = failure_mode

    # ─────────────────────────────────────────
    # Computed properties
    # ─────────────────────────────────────────

    def duration(self) -> float:
        return round((self.finished_at or time.time()) - self.started_at, 3)

    def total_llm_tokens(self) -> int:
        return sum(c.total_tokens() for c in self.llm_calls)

    def total_tool_calls(self) -> int:
        return len(self.tool_calls)

    def failed_tools(self) -> List[str]:
        return [c.tool_name for c in self.tool_calls if not c.success]

    def succeeded(self) -> bool:
        return self.status == EpisodeStatus.SUCCESS

    def avg_causal_confidence(self) -> float:
        if not self.causal_trace:
            return 0.0
        return round(sum(s.confidence for s in self.causal_trace) / len(self.causal_trace), 3)

    # ─────────────────────────────────────────
    # Serialization
    # ─────────────────────────────────────────

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExperienceEpisode":
        if data.get("model_before"):
            data["model_before"] = ModelSnapshot(**data["model_before"])
        if data.get("model_after"):
            data["model_after"]  = ModelSnapshot(**data["model_after"])
        if data.get("evaluation"):
            data["evaluation"]   = EvaluationResult(**data["evaluation"])
        if data.get("tool_calls"):
            data["tool_calls"]   = [ToolCallRecord(**t) for t in data["tool_calls"]]
        if data.get("llm_calls"):
            data["llm_calls"]    = [LLMCallRecord(**c) for c in data["llm_calls"]]
        if data.get("causal_trace"):
            data["causal_trace"] = [CausalStep(**s) for s in data["causal_trace"]]

        data["task_type"]    = TaskType(data.get("task_type", "unknown"))
        data["status"]       = EpisodeStatus(data.get("status", "running"))
        data["failure_mode"] = FailureMode(data.get("failure_mode", "none"))

        # safe defaults for AGI fields (backward compat with old saved episodes)
        data.setdefault("causal_trace", [])
        data.setdefault("counterfactual_options", [])
        data.setdefault("uncertainty_score", 0.5)
        data.setdefault("parent_episode_id", None)

        return cls(**data)

    def save(self, base_dir: str = "experiments/episodes") -> Path:
        path = Path(base_dir)
        path.mkdir(parents=True, exist_ok=True)
        file_path = path / f"{self.episode_id}.json"
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, default=str)
        return file_path

    @classmethod
    def load(cls, file_path: str) -> "ExperienceEpisode":
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)

    @classmethod
    def load_all(cls, base_dir: str = "experiments/episodes") -> List["ExperienceEpisode"]:
        path = Path(base_dir)
        if not path.exists():
            return []
        episodes = []
        for json_file in sorted(path.glob("*.json")):
            if json_file.name.startswith("_"):
                continue  # skip _index.json
            try:
                episodes.append(cls.load(str(json_file)))
            except Exception:
                pass
        episodes.sort(key=lambda e: e.started_at)
        return episodes

    # ─────────────────────────────────────────
    # Display
    # ─────────────────────────────────────────

    def summary(self) -> str:
        dur   = f"{self.duration():.1f}s"
        acc   = f" | acc={self.evaluation.accuracy:.2f}" if self.evaluation and self.evaluation.accuracy is not None else ""
        cost  = f" | ${self.cost_usd:.5f}" if self.cost_usd > 0 else ""
        genes = f" | {len(self.gene_hints)}g" if self.gene_hints else ""
        cf    = f" | {len(self.counterfactual_options)}cf" if self.counterfactual_options else ""
        u     = f" | u={self.uncertainty_score:.2f}"
        return (
            f"[{self.episode_id[:8]}] {self.task_type.value} | "
            f"{self.status.value} | {dur}{acc}{cost}{genes}{cf}{u}"
        )

    def full_report(self) -> str:
        lines = [
            "=" * 65,
            f"Episode ID    : {self.episode_id}",
            f"Task          : {self.task_description}",
            f"Type          : {self.task_type.value}",
            f"Status        : {self.status.value}",
            f"Failure Mode  : {self.failure_mode.value}",
            f"Uncertainty   : {self.uncertainty_score:.2f}",
            f"Duration      : {self.duration():.3f}s",
            f"Cost (USD)    : ${self.cost_usd:.6f}",
            f"Parent Ep     : {self.parent_episode_id or 'none'}",
        ]

        if self.causal_trace:
            lines.append("\n── Causal Trace ─────────────────────────────")
            for cs in self.causal_trace:
                lines.append(f"  [{cs.confidence:.2f}] {cs.decision} ← {cs.reason}")

        if self.counterfactual_options:
            lines.append("\n── Counterfactuals (rejected options) ───────")
            for cf in self.counterfactual_options:
                lines.append(f"  ✗ {cf}")

        lines.append("\n── Tool Calls ───────────────────────────────")
        for tc in self.tool_calls:
            icon = "✓" if tc.success else "✗"
            lines.append(f"  {icon} {tc.tool_name}: {tc.output_summary[:70]}")

        if self.evaluation:
            ev = self.evaluation
            lines += [
                "\n── Evaluation ───────────────────────────────",
                f"  Accuracy      : {ev.accuracy}",
                f"  Hallucination : {ev.hallucination_rate}",
                f"  Safety        : {ev.safety_score}",
                f"  Plan Eff.     : {ev.plan_efficiency}",
            ]

        if self.gene_hints:
            lines.append("\n── Gene Hints ───────────────────────────────")
            for g in self.gene_hints:
                lines.append(f"  💡 {g}")

        lines += [
            "\n── Reflection ───────────────────────────────",
            f"  {self.agent_reflection or '(none)'}",
            f"  Next: {self.next_improvement or '(none)'}",
            "=" * 65,
        ]
        return "\n".join(lines)


# ─────────────────────────────────────────────
# Adds: persistent index, multi-filter search, statistics, pruning
# ─────────────────────────────────────────────

class EpisodeStore:

    def __init__(self, base_dir: str = "experiments/episodes") -> None:
        self.base_dir    = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self._index_file = self.base_dir / "_index.json"
        self._index: Dict[str, Dict] = self._load_index()

    def _load_index(self) -> Dict[str, Dict]:
        if self._index_file.exists():
            try:
                return json.loads(self._index_file.read_text(encoding="utf-8"))
            except Exception:
                pass
        return {}

    def _save_index(self) -> None:
        self._index_file.write_text(
            json.dumps(self._index, indent=2, default=str), encoding="utf-8"
        )

    def _index_episode(self, ep: ExperienceEpisode) -> None:
        acc = ep.evaluation.accuracy if ep.evaluation and ep.evaluation.accuracy is not None else 0.0
        self._index[ep.episode_id] = {
            "episode_id":   ep.episode_id,
            "started_at":   ep.started_at,
            "task":         ep.task_description,
            "task_type":    ep.task_type.value,
            "status":       ep.status.value,
            "failure_mode": ep.failure_mode.value,
            "accuracy":     acc,
            "uncertainty":  ep.uncertainty_score,
            "cost_usd":     ep.cost_usd,
            "tags":         ep.tags,
            "has_parent":   ep.parent_episode_id is not None,
            "gene_count":   len(ep.gene_hints),
            "causal_steps": len(ep.causal_trace),
        }
        self._save_index()

    def save(self, ep: ExperienceEpisode) -> Path:
        path = ep.save(str(self.base_dir))
        self._index_episode(ep)
        return path

    def load(self, episode_id: str) -> Optional[ExperienceEpisode]:
        path = self.base_dir / f"{episode_id}.json"
        if not path.exists():
            return None
        try:
            return ExperienceEpisode.load(str(path))
        except Exception:
            return None

    def load_all(self) -> List[ExperienceEpisode]:
        out = []
        for ep_id in self._index:
            ep = self.load(ep_id)
            if ep:
                out.append(ep)
        return out

    def delete(self, episode_id: str) -> None:
        path = self.base_dir / f"{episode_id}.json"
        if path.exists():
            path.unlink()
        self._index.pop(episode_id, None)
        self._save_index()

    def search(
        self,
        task_keywords: Optional[List[str]]    = None,
        task_type: Optional[TaskType]          = None,
        status: Optional[EpisodeStatus]        = None,
        failure_mode: Optional[FailureMode]    = None,
        min_accuracy: Optional[float]          = None,
        max_uncertainty: Optional[float]       = None,
        success_only: bool                     = False,
        has_causal_trace: Optional[bool]       = None,
        tags: Optional[List[str]]              = None,
        limit: Optional[int]                   = None,
    ) -> List[ExperienceEpisode]:
        results = []
        for ep_id, idx in self._index.items():
            if success_only and idx["status"] != "success":
                continue
            if task_type and idx["task_type"] != task_type.value:
                continue
            if status and idx["status"] != status.value:
                continue
            if failure_mode and idx["failure_mode"] != failure_mode.value:
                continue
            if min_accuracy is not None and idx["accuracy"] < min_accuracy:
                continue
            if max_uncertainty is not None and idx["uncertainty"] > max_uncertainty:
                continue
            if has_causal_trace is not None:
                if has_causal_trace and idx["causal_steps"] == 0:
                    continue
                if not has_causal_trace and idx["causal_steps"] > 0:
                    continue
            if tags:
                if not any(t in idx.get("tags", []) for t in tags):
                    continue
            if task_keywords:
                task_lower = idx["task"].lower()
                if not all(k.lower() in task_lower for k in task_keywords):
                    continue
            ep = self.load(ep_id)
            if ep:
                results.append(ep)

        results.sort(key=lambda e: e.started_at, reverse=True)
        return results[:limit] if limit else results

    def get_statistics(self) -> Dict[str, Any]:
        episodes = self.load_all()
        if not episodes:
            return {"total": 0}
        total      = len(episodes)
        successful = sum(1 for e in episodes if e.status == EpisodeStatus.SUCCESS)
        avg_acc    = sum(
            e.evaluation.accuracy for e in episodes
            if e.evaluation and e.evaluation.accuracy is not None
        ) / max(total, 1)
        failure_dist: Dict[str, int] = {}
        for e in episodes:
            k = e.failure_mode.value
            failure_dist[k] = failure_dist.get(k, 0) + 1
        return {
            "total":            total,
            "successful":       successful,
            "success_rate":     round(successful / total, 3),
            "avg_accuracy":     round(avg_acc, 3),
            "avg_uncertainty":  round(sum(e.uncertainty_score for e in episodes) / total, 3),
            "total_cost_usd":   round(sum(e.cost_usd for e in episodes), 6),
            "failure_modes":    failure_dist,
            "episodes_with_causal": sum(1 for e in episodes if e.causal_trace),
        }

    def prune(self, keep_recent: int = 100, keep_successful: bool = True, dry_run: bool = False) -> List[str]:
        episodes = self.load_all()
        episodes.sort(key=lambda e: e.started_at, reverse=True)
        keep = {ep.episode_id for ep in episodes[:keep_recent]}
        if keep_successful:
            keep |= {ep.episode_id for ep in episodes if ep.status == EpisodeStatus.SUCCESS}
        to_delete = [ep.episode_id for ep in episodes if ep.episode_id not in keep]
        if not dry_run:
            for ep_id in to_delete:
                self.delete(ep_id)
        return to_delete

    def export(self, output_dir: str) -> List[Path]:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        exported = []
        for ep_id in self._index:
            src = self.base_dir / f"{ep_id}.json"
            if src.exists():
                dst = out / src.name
                shutil.copy2(src, dst)
                exported.append(dst)
        return exported

    def get_lineage(self, episode_id: str) -> List[ExperienceEpisode]:
        """Follow parent_episode_id chain — returns full evolution tree for one task."""
        chain = []
        current_id: Optional[str] = episode_id
        visited = set()
        while current_id and current_id not in visited:
            visited.add(current_id)
            ep = self.load(current_id)
            if not ep:
                break
            chain.append(ep)
            current_id = ep.parent_episode_id
        chain.reverse()
        return chain

    def __len__(self) -> int:
        return len(self._index)

    def __iter__(self) -> Iterator[ExperienceEpisode]:
        for ep_id in self._index:
            ep = self.load(ep_id)
            if ep:
                yield ep