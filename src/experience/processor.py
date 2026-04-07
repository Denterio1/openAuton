"""
experience/processor.py
=======================
Unified Self‑Evaluation Processor

Combines:
- Quantitative scoring (model + agent) with letter grades
- Trend analysis (improving/declining/stable)
- Priority actions and next hints
- Causal trace (why decisions were made)
- Counterfactual options (what could have been different)
- Gene hints for DNA evolution
- Cost & time tracking

Enhanced with optional LLM layer for deeper qualitative insights,
including meta‑cognition suggestions for improving the agent itself.
"""

from __future__ import annotations
import asyncio
import json
import math
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any

from .episodes import ExperienceEpisode, EpisodeStatus, EvaluationResult, TaskType

# Optional LLM import
try:
    from llm.provider import ModelProvider
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False
    ModelProvider = None

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────
# Helper functions
# ──────────────────────────────────────────────────────────

def _grade(score: Optional[float]) -> str:
    if score is None:
        return "N/A"
    if score >= 0.90: return "A"
    if score >= 0.80: return "B"
    if score >= 0.70: return "C"
    if score >= 0.60: return "D"
    return "F"


def _trend(past_values: List[Optional[float]], current: Optional[float], lower_is_better: bool = False) -> str:
    valid = [v for v in past_values[-3:] if v is not None]
    if not valid or current is None:
        return "unknown"
    avg = sum(valid) / len(valid)
    delta = current - avg
    if lower_is_better:
        delta = -delta
    if delta > 0.03:
        return "improving ↑"
    if delta < -0.03:
        return "declining ↓"
    return "stable →"


# ──────────────────────────────────────────────────────────
# Unified Report
# ──────────────────────────────────────────────────────────

@dataclass
class UnifiedReport:
    """Complete self‑evaluation report – all metrics + insights."""

    # Basic info
    episode_id: str
    task_type: TaskType
    status: EpisodeStatus

    # Model performance
    accuracy: Optional[float]
    reasoning_quality: Optional[float]
    hallucination_rate: Optional[float]
    safety_score: Optional[float]
    model_score: float
    model_grade: str
    accuracy_ok: bool
    hallucination_ok: bool
    safety_ok: bool

    # Agent performance
    plan_efficiency: float
    tool_usage_correct: bool
    learned_from_history: bool
    agent_score: float
    agent_grade: str
    plan_was_efficient: bool
    tools_were_correct: bool

    # Trends
    accuracy_trend: str
    hallucination_trend: str
    overall_trend: str

    # Actions & hints
    priority_actions: List[str]
    next_architecture_hint: str
    next_data_hint: str
    next_training_hint: str
    next_improvement: str

    # AGI‑level insights
    lessons_learned: List[str]
    gene_hints: List[str]
    causal_trace: List[Dict[str, str]]
    counterfactual_options: List[str]

    # Cost & time
    total_duration_seconds: float
    token_cost_estimate: float
    gpu_hours_estimate: float

    # One‑line summary
    one_line: str

    # Meta‑suggestions (new)
    meta_suggestions: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "task_type": self.task_type.value,
            "status": self.status.value,
            "accuracy": self.accuracy,
            "reasoning_quality": self.reasoning_quality,
            "hallucination_rate": self.hallucination_rate,
            "safety_score": self.safety_score,
            "model_score": self.model_score,
            "model_grade": self.model_grade,
            "accuracy_ok": self.accuracy_ok,
            "hallucination_ok": self.hallucination_ok,
            "safety_ok": self.safety_ok,
            "plan_efficiency": self.plan_efficiency,
            "tool_usage_correct": self.tool_usage_correct,
            "learned_from_history": self.learned_from_history,
            "agent_score": self.agent_score,
            "agent_grade": self.agent_grade,
            "plan_was_efficient": self.plan_was_efficient,
            "tools_were_correct": self.tools_were_correct,
            "accuracy_trend": self.accuracy_trend,
            "hallucination_trend": self.hallucination_trend,
            "overall_trend": self.overall_trend,
            "priority_actions": self.priority_actions,
            "next_architecture_hint": self.next_architecture_hint,
            "next_data_hint": self.next_data_hint,
            "next_training_hint": self.next_training_hint,
            "next_improvement": self.next_improvement,
            "lessons_learned": self.lessons_learned,
            "gene_hints": self.gene_hints,
            "causal_trace": self.causal_trace,
            "counterfactual_options": self.counterfactual_options,
            "total_duration_seconds": self.total_duration_seconds,
            "token_cost_estimate": self.token_cost_estimate,
            "gpu_hours_estimate": self.gpu_hours_estimate,
            "one_line": self.one_line,
            "meta_suggestions": self.meta_suggestions,
        }

    def full_report(self) -> str:
        lines = [
            "=" * 60,
            f"Unified Self‑Evaluation Report [{self.episode_id[:8]}]",
            "",
            "── Model Performance ────────────────────",
            f"  Score      : {self.model_score:.2f}  ({self.model_grade})",
            f"  Accuracy   : {(self.accuracy or 0):.3f}  {'✅' if self.accuracy_ok else '❌'}",
            f"  Hallucin.  : {(self.hallucination_rate or 0):.3f}  {'✅' if self.hallucination_ok else '❌'}",
            f"  Safety     : {(self.safety_score or 0):.3f}  {'✅' if self.safety_ok else '❌'}",
            "",
            "── Agent Performance ────────────────────",
            f"  Score      : {self.agent_score:.2f}  ({self.agent_grade})",
            f"  Plan eff.  : {(self.plan_efficiency or 0):.2f}",
            f"  Tools ok   : {'✅' if self.tool_usage_correct else '❌'}",
            f"  Used hist. : {'✅' if self.learned_from_history else '❌'}",
            "",
            "── Trends ───────────────────────────────",
            f"  Accuracy   : {self.accuracy_trend}",
            f"  Hallucin.  : {self.hallucination_trend}",
            f"  Overall    : {self.overall_trend}",
            "",
            "── Priority Actions ─────────────────────",
        ]
        for i, action in enumerate(self.priority_actions, 1):
            lines.append(f"  {i}. {action}")
        lines += [
            "",
            "── Next Hints ───────────────────────────",
            f"  Architecture : {self.next_architecture_hint}",
            f"  Data         : {self.next_data_hint}",
            f"  Training     : {self.next_training_hint}",
            "",
            "── AGI Insights ─────────────────────────",
            f"  Lessons      : {', '.join(self.lessons_learned[:2])}",
            f"  Gene hints   : {', '.join(self.gene_hints[:3])}",
            f"  Counterfactuals: {', '.join(self.counterfactual_options[:2])}",
        ]
        if self.meta_suggestions:
            lines += ["", "── Meta‑Suggestions (Agent Improvement) ───"]
            for sug in self.meta_suggestions[:3]:
                lines.append(f"  • {sug.get('component')}: {sug.get('reasoning', '')[:80]}")
        lines += [
            "",
            f"Summary: {self.one_line}",
            "=" * 60,
        ]
        return "\n".join(lines)


# ──────────────────────────────────────────────────────────
# The Processor Class
# ──────────────────────────────────────────────────────────

class ReflectionProcessor:
    """
    Unified processor that combines quantitative scoring + qualitative reflection.
    Optionally uses an LLM to generate deeper insights and meta‑suggestions.
    """

    ACC_TARGET = 0.80
    HALLU_MAX = 0.10
    SAFETY_MIN = 0.90
    RQ_TARGET = 0.70

    def __init__(self, llm_provider=None):
        self.llm = llm_provider
        self._eval_count = 0

    # ------------------------------------------------------------------
    # Main sync method (rule‑based only)
    # ------------------------------------------------------------------

    def process(self, episode: ExperienceEpisode, history: List[ExperienceEpisode]) -> UnifiedReport:
        return self._build_report(episode, history, use_llm=False)

    # ------------------------------------------------------------------
    # Async method with LLM enhancement
    # ------------------------------------------------------------------

    async def process_async(self, episode: ExperienceEpisode, history: List[ExperienceEpisode]) -> UnifiedReport:
        return await self._build_report_async(episode, history, use_llm=(self.llm is not None))

    # ------------------------------------------------------------------
    # Internal builders
    # ------------------------------------------------------------------

    def _build_report(self, episode: ExperienceEpisode, history: List[ExperienceEpisode],
                      use_llm: bool = False) -> UnifiedReport:
        base = self._compute_base_report(episode, history)
        if use_llm and self.llm:
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                enhanced = loop.run_until_complete(self._llm_enhance(episode, history, base))
                loop.close()
                base.priority_actions = enhanced.get("priority_actions", base.priority_actions)
                base.lessons_learned = enhanced.get("lessons_learned", base.lessons_learned)
                base.gene_hints = enhanced.get("gene_hints", base.gene_hints)
                base.next_improvement = enhanced.get("next_improvement", base.next_improvement)
                # New: meta‑suggestions
                meta = enhanced.get("meta_suggestions", [])
                if meta:
                    base.meta_suggestions = meta
            except Exception as e:
                logger.warning(f"LLM enhancement failed: {e}, using rule‑based.")
        return base

    async def _build_report_async(self, episode: ExperienceEpisode, history: List[ExperienceEpisode],
                                   use_llm: bool = False) -> UnifiedReport:
        base = self._compute_base_report(episode, history)
        if use_llm and self.llm:
            try:
                enhanced = await self._llm_enhance(episode, history, base)
                base.priority_actions = enhanced.get("priority_actions", base.priority_actions)
                base.lessons_learned = enhanced.get("lessons_learned", base.lessons_learned)
                base.gene_hints = enhanced.get("gene_hints", base.gene_hints)
                base.next_improvement = enhanced.get("next_improvement", base.next_improvement)
                if "meta_suggestions" in enhanced:
                    base.meta_suggestions = enhanced["meta_suggestions"]
            except Exception as e:
                logger.warning(f"LLM enhancement failed: {e}, using rule‑based.")
        return base

    # ------------------------------------------------------------------
    # Quantitative computation (unchanged)
    # ------------------------------------------------------------------

    def _compute_base_report(self, episode: ExperienceEpisode, history: List[ExperienceEpisode]) -> UnifiedReport:
        ev = episode.evaluation
        if ev is None:
            raise ValueError("Episode has no evaluation – cannot process reflection.")

        acc = ev.accuracy
        rq = ev.reasoning_quality
        hallu = ev.hallucination_rate
        safety = ev.safety_score

        acc_ok = acc is not None and acc >= self.ACC_TARGET
        hallu_ok = hallu is not None and hallu <= self.HALLU_MAX
        safety_ok = safety is not None and safety >= self.SAFETY_MIN

        model_score = self._model_score(acc, hallu, safety, rq)
        model_grade = _grade(model_score)

        plan_ok = ev.plan_efficiency >= 0.70 if ev.plan_efficiency is not None else False
        tools_ok = ev.tool_usage_correct if ev.tool_usage_correct is not None else True
        hist_ok = ev.learned_from_history if ev.learned_from_history is not None else False

        agent_score = self._agent_score(plan_ok, tools_ok, hist_ok, episode)
        agent_grade = _grade(agent_score)

        same_type = [
            e for e in history
            if e.task_type == episode.task_type
            and e.evaluation is not None
            and e.episode_id != episode.episode_id
        ]
        acc_trend = _trend([e.evaluation.accuracy for e in same_type], acc)
        hallu_trend = _trend([e.evaluation.hallucination_rate for e in same_type], hallu, lower_is_better=True)
        overall_trend = self._overall_trend(model_score, same_type)

        actions = self._priority_actions(acc_ok, hallu_ok, safety_ok, rq, plan_ok, tools_ok, hist_ok)
        arch_hint = self._arch_hint(acc_ok, rq)
        data_hint = self._data_hint(hallu_ok, acc_ok)
        train_hint = self._train_hint(hallu_ok, rq, safety_ok)

        lessons = []
        if acc is not None and acc < 0.7:
            lessons.append("Accuracy too low → increase model capacity or add more data.")
        if hallu is not None and hallu > 0.15:
            lessons.append("Hallucination above threshold → consider RLHF or contrastive loss.")
        if safety is not None and safety < 0.85:
            lessons.append("Safety score low → fine-tune on safe data or add input filtering.")
        if not hist_ok:
            lessons.append("Did not use past episodes – enable retrieval to improve.")

        if lessons:
            next_improvement = " | ".join(lessons[:2])
        else:
            next_improvement = "All metrics acceptable. Try increasing complexity or exploring new architectures."

        gene_hints = []
        if acc is not None and acc > 0.75:
            gene_hints.append(f"high_accuracy_{int(acc*100)}")
        if hallu is not None and hallu < 0.1:
            gene_hints.append("low_hallucination")
        if safety is not None and safety > 0.9:
            gene_hints.append("high_safety")
        if hist_ok:
            gene_hints.append("learned_from_history")

        causal_trace = []
        if not tools_ok:
            causal_trace.append({"decision": "tool selection", "reason": "some tools failed, need fallback"})
        if not hist_ok:
            causal_trace.append({"decision": "retrieval", "reason": "history was ignored or unavailable"})

        counterfactuals = []
        if acc is not None and acc < 0.7:
            counterfactuals.append("Use deeper transformer or more epochs")
        if hallu is not None and hallu > 0.15:
            counterfactuals.append("Add chain-of-thought fine-tuning")

        duration = getattr(episode, "duration_seconds", 0.0)
        token_cost = getattr(episode, "token_cost_estimate", 0.0)
        gpu_hours = 0.0
        one_line = f"{model_grade} model / {agent_grade} agent — " + (actions[0] if actions else "all targets met")

        return UnifiedReport(
            episode_id=episode.episode_id,
            task_type=episode.task_type,
            status=episode.status,
            accuracy=acc,
            reasoning_quality=rq,
            hallucination_rate=hallu,
            safety_score=safety,
            model_score=model_score,
            model_grade=model_grade,
            accuracy_ok=acc_ok,
            hallucination_ok=hallu_ok,
            safety_ok=safety_ok,
            plan_efficiency=ev.plan_efficiency if ev.plan_efficiency is not None else 0.5,
            tool_usage_correct=tools_ok,
            learned_from_history=hist_ok,
            agent_score=agent_score,
            agent_grade=agent_grade,
            plan_was_efficient=plan_ok,
            tools_were_correct=tools_ok,
            accuracy_trend=acc_trend,
            hallucination_trend=hallu_trend,
            overall_trend=overall_trend,
            priority_actions=actions,
            next_architecture_hint=arch_hint,
            next_data_hint=data_hint,
            next_training_hint=train_hint,
            next_improvement=next_improvement,
            lessons_learned=lessons,
            gene_hints=gene_hints,
            causal_trace=causal_trace,
            counterfactual_options=counterfactuals,
            total_duration_seconds=duration,
            token_cost_estimate=token_cost,
            gpu_hours_estimate=gpu_hours,
            one_line=one_line,
            meta_suggestions=[],
        )

    # ------------------------------------------------------------------
    # LLM enhancement (basic qualitative insights)
    # ------------------------------------------------------------------

    async def _llm_enhance(self, episode: ExperienceEpisode, history: List[ExperienceEpisode],
                           base_report: UnifiedReport) -> Dict[str, Any]:
        if not self.llm:
            return {}

        task_desc = episode.task_description[:200]
        metrics = (f"acc={base_report.accuracy or 0:.2f}, "
                   f"hallu={base_report.hallucination_rate or 0:.2f}, "
                   f"safety={base_report.safety_score or 0:.2f}, "
                   f"reasoning={base_report.reasoning_quality or 0:.2f}")
        model_grade = base_report.model_grade
        agent_grade = base_report.agent_grade
        current_actions = "\n".join(f"- {a}" for a in base_report.priority_actions[:3])
        current_lessons = "\n".join(f"- {l}" for l in base_report.lessons_learned[:2])
        current_hints = "\n".join(f"- {h}" for h in base_report.gene_hints[:3])

        prompt = f"""You are an expert AI engineer reflecting on a training episode.

Task: {task_desc}
Metrics: {metrics}
Model Grade: {model_grade}, Agent Grade: {agent_grade}

Current rule‑based suggestions:
Priority Actions:
{current_actions}

Lessons Learned:
{current_lessons}

Gene Hints:
{current_hints}

Please provide enhanced, more specific, actionable insights.
Return ONLY valid JSON with exactly these keys:
{{
  "priority_actions": ["action1", "action2", ...],
  "lessons_learned": ["lesson1", "lesson2", ...],
  "gene_hints": ["hint1", "hint2", ...],
  "next_improvement": "short actionable summary",
  "meta_suggestions": [
    {{"component": "planner/tools/strategy/dna", "change": {{}}, "reasoning": "...", "confidence": 0.8}}
  ]
}}
Do not include any other text outside the JSON.
"""
        try:
            # Use think() method (adjust to your provider's API)
            resp = self.llm.think(prompt, system="You are a helpful AI that outputs only valid JSON.")
            content = resp.content.strip()
            # Remove markdown fences
            if content.startswith("```json"):
                content = content[7:]
            if content.startswith("```"):
                content = content[3:]
            if content.endswith("```"):
                content = content[:-3]
            data = json.loads(content)
            # Ensure expected keys exist
            for key in ["priority_actions", "lessons_learned", "gene_hints", "next_improvement"]:
                if key not in data:
                    data[key] = []
            return data
        except Exception as e:
            logger.warning(f"LLM enhancement failed: {e}")
            return {}

    # ------------------------------------------------------------------
    # Scoring helpers (unchanged)
    # ------------------------------------------------------------------

    def _model_score(self, acc: Optional[float], hallu: Optional[float], safety: Optional[float], rq: Optional[float]) -> float:
        score = 0.0
        total_w = 0.0
        if acc is not None:
            score += 0.40 * acc; total_w += 0.40
        if hallu is not None:
            score += 0.30 * (1 - hallu); total_w += 0.30
        if safety is not None:
            score += 0.20 * safety; total_w += 0.20
        if rq is not None:
            score += 0.10 * rq; total_w += 0.10
        return score / total_w if total_w > 0 else 0.0

    def _agent_score(self, plan_ok: bool, tools_ok: bool, hist_ok: bool, episode: ExperienceEpisode) -> float:
        score = 0.0
        if plan_ok:   score += 0.40
        if tools_ok:  score += 0.35
        if hist_ok:   score += 0.25
        if episode.status == EpisodeStatus.SUCCESS:
            score = min(1.0, score + 0.10)
        return score

    def _overall_trend(self, current_model_score: float, history: List[ExperienceEpisode]) -> str:
        past_scores = []
        for e in history[-3:]:
            if e.evaluation:
                s = self._model_score(
                    e.evaluation.accuracy,
                    e.evaluation.hallucination_rate,
                    e.evaluation.safety_score,
                    e.evaluation.reasoning_quality,
                )
                past_scores.append(s)
        if not past_scores:
            return "unknown"
        avg = sum(past_scores) / len(past_scores)
        delta = current_model_score - avg
        if delta > 0.03:  return "improving ↑"
        if delta < -0.03: return "declining ↓"
        return "stable →"

    def _priority_actions(self, acc_ok: bool, hallu_ok: bool, safety_ok: bool, rq: Optional[float],
                          plan_ok: bool, tools_ok: bool, hist_ok: bool) -> List[str]:
        actions = []
        if not acc_ok:
            actions.append("Increase model depth (add 2 layers) or double training data")
        if not hallu_ok:
            actions.append("Add RLHF / contrastive loss to reduce hallucination")
        if not safety_ok:
            actions.append("Run extended safety fine-tuning before deployment")
        if rq is not None and rq < self.RQ_TARGET:
            actions.append("Introduce chain-of-thought supervision loss term")
        if not plan_ok:
            actions.append("Planner: tighten step selection — avoid redundant tool calls")
        if not tools_ok:
            actions.append("Some tools failed — check tool configs or add fallbacks")
        if not hist_ok:
            actions.append("Enable history retrieval — agent is not learning from past")
        if not actions:
            actions.append("All targets met — try harder benchmark or larger model")
        return actions

    def _arch_hint(self, acc_ok: bool, rq: Optional[float]) -> str:
        if not acc_ok:
            return "Increase num_layers by 2 and d_model to 512"
        if rq is not None and rq < self.RQ_TARGET:
            return "Add cross-attention layer for reasoning tasks"
        return "Architecture is sufficient — keep current config"

    def _data_hint(self, hallu_ok: bool, acc_ok: bool) -> str:
        if not hallu_ok:
            return "Increase data diversity score; add contrastive negative examples"
        if not acc_ok:
            return "Increase n_samples by 50% and quality_score filter"
        return "Data quality is adequate"

    def _train_hint(self, hallu_ok: bool, rq: Optional[float], safety_ok: bool) -> str:
        hints = []
        if not hallu_ok:
            hints.append("halve learning rate")
        if rq is not None and rq < self.RQ_TARGET:
            hints.append("add CoT loss term, increase epochs by 5")
        if not safety_ok:
            hints.append("add safety alignment phase after main training")
        return "; ".join(hints) if hints else "Training config is adequate"

    def stats(self) -> str:
        return f"ReflectionProcessor: {self._eval_count} evaluations processed."


# ──────────────────────────────────────────────────────────
# Convenience functions
# ──────────────────────────────────────────────────────────

def process_episode(episode: ExperienceEpisode, history: List[ExperienceEpisode]) -> UnifiedReport:
    proc = ReflectionProcessor()
    return proc.process(episode, history)


async def process_episode_async(episode: ExperienceEpisode, history: List[ExperienceEpisode],
                                 llm_provider=None) -> UnifiedReport:
    proc = ReflectionProcessor(llm_provider=llm_provider)
    return await proc.process_async(episode, history)