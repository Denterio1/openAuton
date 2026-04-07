"""
core/intuition.py
=================
Intuition Engine – filtering layer between Planner and Executor.

Uses cognitive DNA, past episode failure patterns, and cost models to:
- Score each plan step with intuition_score (0..1)
- Predict likely failure type and suggest fallback
- Estimate financial and time cost before execution

Prevents the agent from wasting resources on plans likely to fail.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Tuple
from pathlib import Path
import random
import math
from datetime import datetime

# Import from project
from ..experience.episodes import ExperienceEpisode, TaskType, EpisodeStatus, EvaluationResult
from ..genome.dna import CognitiveDNA, Gene, GeneType


# ──────────────────────────────────────────────────────────
# Data structures for intuition results
# ──────────────────────────────────────────────────────────

@dataclass
class IntuitionScore:
    """Score for a single plan step."""
    step_index: int
    step_tool: str
    intuition_score: float          # 0..1 – higher means more confident it will succeed
    reason: str
    predicted_risk: Optional[str]   # "high_cost", "hallucination", "memory_overflow", "timeout", etc.
    fallback_step: Optional[str]    # alternative tool or parameter change

@dataclass
class PlanIntuition:
    """Overall intuition for a complete plan."""
    step_scores: List[IntuitionScore]
    overall_score: float            # average of step scores weighted by criticality
    estimated_cost_usd: float
    estimated_time_seconds: float
    recommended: bool               # True if overall_score > threshold (e.g., 0.6)
    warning: Optional[str]

    def summary(self) -> str:
        lines = [
            f"Intuition Summary: overall={self.overall_score:.2f} {'✅ recommended' if self.recommended else '❌ NOT recommended'}",
            f"  Est. cost: ${self.estimated_cost_usd:.4f} | Est. time: {self.estimated_time_seconds:.1f}s",
        ]
        if self.warning:
            lines.append(f"  ⚠️ Warning: {self.warning}")
        for step in self.step_scores:
            emoji = "✅" if step.intuition_score > 0.7 else "⚠️" if step.intuition_score > 0.4 else "❌"
            lines.append(f"    {emoji} Step {step.step_index}: {step.step_tool} ({step.intuition_score:.2f}) – {step.reason}")
        return "\n".join(lines)


# ──────────────────────────────────────────────────────────
# 1. Feasibility Predictor
# ──────────────────────────────────────────────────────────

class FeasibilityPredictor:
    """
    Assigns intuition_score to each plan step by comparing with past failure patterns
    and current cognitive DNA.
    """

    def __init__(self, history: List[ExperienceEpisode], dna: Optional[CognitiveDNA] = None):
        self.history = history
        self.dna = dna

    def score_step(self, step_index: int, tool_name: str, tool_params: Dict, task_type: TaskType) -> IntuitionScore:
        """
        Calculate intuition score based on:
        - Historical success/failure rate of this tool for similar task types
        - DNA preferences (e.g., prefers deeper networks, avoids certain tools)
        - Parameter sanity (e.g., too large batch size → lower score)
        """
        base_score = 0.5
        reason_parts = []
        risk = None
        fallback = None

        # 1. Historical success rate for this tool (similar task type)
        relevant_episodes = [ep for ep in self.history if ep.task_type == task_type]
        tool_calls = []
        for ep in relevant_episodes:
            for call in ep.tool_calls:
                if call.tool_name == tool_name:
                    tool_calls.append(call)
        if tool_calls:
            success_count = sum(1 for c in tool_calls if c.success)
            success_rate = success_count / len(tool_calls)
            base_score = base_score * 0.5 + success_rate * 0.5
            reason_parts.append(f"historical success rate {success_rate:.0%}")
        else:
            reason_parts.append("no history – neutral")

        # 2. DNA influence – certain genes may increase confidence
        if self.dna:
            for gene in self.dna.genes:
                if gene.gene_type == GeneType.META_STRATEGY and gene.name == "use_history":
                    # gene with high confidence encourages using tools that were successful
                    if tool_calls and success_rate > 0.7:
                        base_score = min(1.0, base_score + gene.confidence * 0.1)
                        reason_parts.append(f"DNA gene '{gene.name}' boosts confidence")
                elif gene.gene_type == GeneType.ARCHITECTURE and gene.name == "prefer_deeper_network":
                    if tool_name == "run_training_job" and tool_params.get("num_layers", 0) < 6:
                        base_score = max(0.0, base_score - 0.1)
                        reason_parts.append("DNA prefers deeper networks, current shallow")

        # 3. Parameter sanity checks
        if tool_name == "run_training_job":
            epochs = tool_params.get("epochs", 10)
            if epochs > 100:
                base_score *= 0.7
                reason_parts.append("too many epochs → cost risk")
                risk = "high_cost"
                fallback = f"reduce epochs to {epochs // 2}"
            batch_size = tool_params.get("batch_size", 32)
            if batch_size > 128:
                base_score *= 0.8
                reason_parts.append("large batch size may cause OOM")
                risk = "memory_overflow"

        elif tool_name == "evaluate_model":
            # usually safe
            pass
        elif tool_name == "generate_test_cases":
            num_cases = tool_params.get("num_cases", 100)
            if num_cases > 500:
                base_score *= 0.9
                reason_parts.append("many test cases → time cost")

        # Clamp score
        intuition_score = max(0.0, min(1.0, base_score))
        reason = " | ".join(reason_parts) if reason_parts else "no specific signals"

        return IntuitionScore(
            step_index=step_index,
            step_tool=tool_name,
            intuition_score=intuition_score,
            reason=reason,
            predicted_risk=risk,
            fallback_step=fallback,
        )


# ──────────────────────────────────────────────────────────
# 2. Risk Anticipator
# ──────────────────────────────────────────────────────────

class RiskAnticipator:
    """
    Predicts likely failure type before execution, based on historical failure patterns.
    """

    # Common failure patterns extracted from episodes
    FAILURE_PATTERNS = {
        "high_cost": ["run_training_job", "generate_test_cases"],
        "hallucination": ["run_training_job", "evaluate_model"],
        "memory_overflow": ["run_training_job"],
        "timeout": ["web_search", "python_executor"],
        "safety_violation": ["run_safety_check"],
    }

    def __init__(self, history: List[ExperienceEpisode]):
        self.history = history
        # Build failure frequency map
        self.failure_map = {}
        for ep in history:
            if ep.status != EpisodeStatus.SUCCESS and ep.evaluation:
                for call in ep.tool_calls:
                    if not call.success:
                        tool = call.tool_name
                        if tool not in self.failure_map:
                            self.failure_map[tool] = {}
                        # Try to infer failure type from evaluation notes
                        if ep.evaluation.notes:
                            if "hallucination" in ep.evaluation.notes.lower():
                                self.failure_map[tool]["hallucination"] = self.failure_map[tool].get("hallucination", 0) + 1
                            if "cost" in ep.evaluation.notes.lower() or "expensive" in ep.evaluation.notes.lower():
                                self.failure_map[tool]["high_cost"] = self.failure_map[tool].get("high_cost", 0) + 1

    def predict_risk(self, tool_name: str, tool_params: Dict) -> Tuple[Optional[str], Optional[str]]:
        """
        Returns (predicted_risk, suggested_fallback).
        """
        # Check against built‑in patterns
        for risk_type, tools in self.FAILURE_PATTERNS.items():
            if tool_name in tools:
                # Additional heuristics
                if risk_type == "high_cost" and tool_name == "run_training_job":
                    epochs = tool_params.get("epochs", 10)
                    if epochs > 50:
                        return "high_cost", f"reduce epochs to {epochs // 2}"
                if risk_type == "memory_overflow" and tool_name == "run_training_job":
                    batch = tool_params.get("batch_size", 32)
                    if batch > 64:
                        return "memory_overflow", f"reduce batch_size to {batch // 2}"
                if risk_type == "hallucination" and tool_name == "run_training_job":
                    # If history shows hallucination problems, suggest RLHF
                    if self.failure_map.get(tool_name, {}).get("hallucination", 0) > 2:
                        return "hallucination", "add RLHF or contrastive loss"

        # Check learned failure map
        if tool_name in self.failure_map:
            # Most frequent risk for this tool
            risks = self.failure_map[tool_name]
            if risks:
                most_common = max(risks, key=risks.get)
                return most_common, f"avoid previous failure: {most_common}"

        return None, None


# ──────────────────────────────────────────────────────────
# 3. Mental Cost Simulator
# ──────────────────────────────────────────────────────────

class MentalCostSimulator:
    """
    Estimates USD cost and time for a plan using:
    - Tool‑specific base costs (from config/costs.yaml)
    - Historical durations from past episodes
    - Parameter scaling (e.g., epochs, batch size)
    """

    # Default cost estimates (USD) per tool call
    DEFAULT_TOOL_COSTS = {
        "run_training_job": 0.05,      # base, scales with epochs
        "evaluate_model": 0.01,
        "generate_test_cases": 0.002,
        "run_safety_check": 0.003,
        "web_search": 0.001,
        "python_executor": 0.0005,
        "file_ops": 0.0001,
        "human_input": 0.0,
    }

    DEFAULT_TOOL_TIMES = {
        "run_training_job": 30.0,      # seconds
        "evaluate_model": 5.0,
        "generate_test_cases": 2.0,
        "run_safety_check": 1.0,
        "web_search": 2.0,
        "python_executor": 1.0,
        "file_ops": 0.1,
        "human_input": 10.0,
    }

    def __init__(self, history: List[ExperienceEpisode], cost_config_path: Optional[Path] = None):
        self.history = history
        self.cost_config = self._load_costs(cost_config_path)
        # Build average durations from history
        self.avg_durations = self._compute_avg_durations()

    def _load_costs(self, path: Optional[Path]) -> Dict:
        if path and path.exists():
            import yaml
            with open(path, 'r') as f:
                return yaml.safe_load(f)
        return self.DEFAULT_TOOL_COSTS

    def _compute_avg_durations(self) -> Dict[str, float]:
        durations = {}
        counts = {}
        for ep in self.history:
            for call in ep.tool_calls:
                tool = call.tool_name
                if call.duration_seconds is not None:
                    durations[tool] = durations.get(tool, 0) + call.duration_seconds
                    counts[tool] = counts.get(tool, 0) + 1
        avg = {}
        for tool, total in durations.items():
            avg[tool] = total / counts[tool]
        return avg

    def estimate_step(self, tool_name: str, tool_params: Dict) -> Tuple[float, float]:
        """
        Returns (cost_usd, time_seconds) for a single step.
        """
        base_cost = self.cost_config.get(tool_name, 0.01)
        base_time = self.avg_durations.get(tool_name, self.DEFAULT_TOOL_TIMES.get(tool_name, 1.0))

        # Scale by parameters
        if tool_name == "run_training_job":
            epochs = tool_params.get("epochs", 10)
            batch = tool_params.get("batch_size", 32)
            # Rough scaling: cost ∝ epochs * (batch/32)
            cost = base_cost * (epochs / 10) * (batch / 32)
            time = base_time * (epochs / 10) * (batch / 32)
        elif tool_name == "generate_test_cases":
            num = tool_params.get("num_cases", 100)
            cost = base_cost * (num / 100)
            time = base_time * (num / 100)
        else:
            cost = base_cost
            time = base_time

        return round(cost, 6), round(time, 1)


# ──────────────────────────────────────────────────────────
# 4. Main Intuition Engine
# ──────────────────────────────────────────────────────────

class IntuitionEngine:
    """
    Main entry point – evaluates a full plan and returns intuition scores,
    risks, cost/time estimates, and a recommendation.
    """

    def __init__(self, history: List[ExperienceEpisode], dna: Optional[CognitiveDNA] = None):
        self.history = history
        self.dna = dna
        self.feasibility = FeasibilityPredictor(history, dna)
        self.risk = RiskAnticipator(history)
        self.cost_sim = MentalCostSimulator(history)

    def evaluate_plan(self,
                      plan_steps: List[Dict],   # each dict: {"tool": str, "params": dict}
                      task_type: TaskType,
                      budget_usd: Optional[float] = None) -> PlanIntuition:
        """
        Evaluate a sequence of plan steps.
        plan_steps: e.g., [{"tool": "run_training_job", "params": {"epochs": 10}}]
        """
        step_scores = []
        total_cost = 0.0
        total_time = 0.0
        warnings = []

        for idx, step in enumerate(plan_steps):
            tool = step.get("tool", "unknown")
            params = step.get("params", {})

            # Score feasibility
            score = self.feasibility.score_step(idx, tool, params, task_type)
            step_scores.append(score)

            # Predict risk and possibly adjust fallback
            risk, fallback = self.risk.predict_risk(tool, params)
            if risk and not score.predicted_risk:
                score.predicted_risk = risk
            if fallback and not score.fallback_step:
                score.fallback_step = fallback

            # Estimate cost and time
            cost, dur = self.cost_sim.estimate_step(tool, params)
            total_cost += cost
            total_time += dur

        # Overall score: average of step scores, but early steps weighted slightly more
        if step_scores:
            weights = [1.0 / (i+1) for i in range(len(step_scores))]
            weighted_sum = sum(s.intuition_score * w for s, w in zip(step_scores, weights))
            overall = weighted_sum / sum(weights)
        else:
            overall = 0.0

        # Check budget
        recommended = overall >= 0.6
        warning = None
        if budget_usd and total_cost > budget_usd:
            recommended = False
            warning = f"Estimated cost ${total_cost:.4f} exceeds budget ${budget_usd:.4f}"
        elif overall < 0.6:
            warning = f"Overall intuition score {overall:.2f} below threshold 0.6"

        return PlanIntuition(
            step_scores=step_scores,
            overall_score=overall,
            estimated_cost_usd=total_cost,
            estimated_time_seconds=total_time,
            recommended=recommended,
            warning=warning,
        )