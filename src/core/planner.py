"""
core/planner.py
===============
Planner — the brain that decides WHAT to do and IN WHAT ORDER.

Before the agent acts, the Planner:
  1. Reads the task + history hints
  2. Scores available tools
  3. Builds an ordered execution plan
  4. Detects if re-planning is needed after failure

Design: ReAct-style — Reason first, then Act.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from experience.episodes import ExperienceEpisode, TaskType
from tools.registry import ToolRegistry


# ─────────────────────────────────────────────
# Plan Step
# ─────────────────────────────────────────────

@dataclass
class PlanStep:
    """One step in the planner's execution plan."""
    order: int
    tool_name: str
    kwargs: Dict[str, Any]
    reason: str                     # why this tool was chosen
    required: bool = True           # if False, skip on failure is OK
    depends_on: List[str] = field(default_factory=list)  # tool names

    def to_tuple(self) -> Tuple[str, Dict[str, Any]]:
        return (self.tool_name, self.kwargs)

    def describe(self) -> str:
        dep = f" (after: {', '.join(self.depends_on)})" if self.depends_on else ""
        req = "REQUIRED" if self.required else "OPTIONAL"
        return f"  {self.order}. [{req}] {self.tool_name}{dep} — {self.reason}"


# ─────────────────────────────────────────────
# Planner Result
# ─────────────────────────────────────────────

@dataclass
class PlannerResult:
    """Full output of one planning cycle."""
    task: str
    task_type: TaskType
    steps: List[PlanStep]
    reasoning: str                  # planner's chain-of-thought
    confidence: float               # 0.0–1.0
    hints_applied: List[str] = field(default_factory=list)

    def to_pipeline(self) -> List[Tuple[str, Dict[str, Any]]]:
        """Convert to format ToolRegistry.run_pipeline() expects."""
        return [s.to_tuple() for s in sorted(self.steps, key=lambda s: s.order)]

    def describe(self) -> str:
        lines = [
            "── Planner Output ───────────────────────",
            f"  Task       : {self.task}",
            f"  Type       : {self.task_type.value}",
            f"  Confidence : {self.confidence:.0%}",
            f"  Reasoning  : {self.reasoning}",
            "",
            "  Steps:",
        ]
        for step in sorted(self.steps, key=lambda s: s.order):
            lines.append(step.describe())
        if self.hints_applied:
            lines.append(f"\n  Hints applied: {len(self.hints_applied)}")
            for h in self.hints_applied:
                lines.append(f"    - {h[:80]}")
        return "\n".join(lines)


# ─────────────────────────────────────────────
# Planner
# ─────────────────────────────────────────────

class Planner:
    """
    Decides which tools to run, in what order, with what args.

    Core logic:
      - Maps TaskType → default tool sequence
      - Applies history hints to modify args
      - Scores confidence based on hint coverage
      - Supports re-planning after partial failure
    """

    # Default tool sequences per task type
    _DEFAULT_SEQUENCES: Dict[TaskType, List[str]] = {
        TaskType.TRAIN_FROM_SCRATCH: [
            "generate_test_cases",
            "run_training_job",
            "evaluate_model",
            "run_safety_check",
        ],
        TaskType.FINE_TUNE: [
            "generate_test_cases",
            "run_training_job",
            "evaluate_model",
            "run_safety_check",
        ],
        TaskType.EVALUATION: [
            "generate_test_cases",
            "evaluate_model",
        ],
        TaskType.ARCHITECTURE_DESIGN: [
            "generate_test_cases",
            "run_training_job",
            "evaluate_model",
        ],
        TaskType.SELF_IMPROVEMENT: [
            "generate_test_cases",
            "run_training_job",
            "evaluate_model",
            "run_safety_check",
        ],
        TaskType.MULTI_AGENT_DESIGN: [
            "generate_test_cases",
            "run_training_job",
            "evaluate_model",
            "run_safety_check",
        ],
        TaskType.TOOL_COMPOSITION: [
            "generate_test_cases",
            "run_training_job",
            "evaluate_model",
        ],
        TaskType.UNKNOWN: [
            "generate_test_cases",
            "run_training_job",
            "evaluate_model",
            "run_safety_check",
        ],
    }

    def __init__(self, registry: ToolRegistry) -> None:
        self.registry = registry
        self._plan_count = 0

    # ─────────────────────────────────────────
    # MAIN PLAN METHOD
    # ─────────────────────────────────────────

    def plan(
        self,
        task: str,
        task_type: TaskType,
        hints: List[str],
        arch_name: str = "transformer-6L-d256",
        epochs: int = 10,
        lr: float = 3e-4,
        n_samples: int = 10_000,
    ) -> PlannerResult:
        """
        Build a full execution plan.

        Parameters
        ----------
        task       : raw task description from user
        task_type  : classified task type
        hints      : next_improvement strings from past episodes
        arch_name  : architecture identifier
        epochs     : training epochs
        lr         : learning rate
        n_samples  : data sample count
        """
        self._plan_count += 1

        # ── Apply hints ───────────────────────
        applied_hints = []
        epochs, lr, n_samples, arch_name = self._apply_hints(
            hints, epochs, lr, n_samples, arch_name, applied_hints
        )

        # ── Get tool sequence ─────────────────
        sequence = self._DEFAULT_SEQUENCES.get(
            task_type, self._DEFAULT_SEQUENCES[TaskType.UNKNOWN]
        )

        # ── Filter to available tools ─────────
        sequence = [t for t in sequence if self.registry.has(t)]

        # ── Build steps ───────────────────────
        steps = self._build_steps(
            sequence, task_type, arch_name, epochs, lr, n_samples
        )

        # ── Reasoning chain ───────────────────
        reasoning = self._reason(task_type, hints, applied_hints, epochs, n_samples)

        # ── Confidence ────────────────────────
        confidence = self._score_confidence(task_type, hints, applied_hints)

        return PlannerResult(
            task=task,
            task_type=task_type,
            steps=steps,
            reasoning=reasoning,
            confidence=confidence,
            hints_applied=applied_hints,
        )

    # ─────────────────────────────────────────
    # RE-PLAN after failure
    # ─────────────────────────────────────────

    def replan(
        self,
        original: PlannerResult,
        failed_step: str,
        error: str,
    ) -> PlannerResult:
        """
        Called when a required step fails.
        Produces a modified plan that skips or replaces the failed step.
        """
        new_steps = []
        for step in original.steps:
            if step.tool_name == failed_step:
                # Mark as optional and add a note
                step.required = False
                step.reason = f"[REPLANNED — was: {step.reason}] Error: {error[:60]}"
            new_steps.append(step)

        return PlannerResult(
            task=original.task,
            task_type=original.task_type,
            steps=new_steps,
            reasoning=f"Replanned after '{failed_step}' failed. Marked as optional.",
            confidence=max(0.1, original.confidence - 0.2),
            hints_applied=original.hints_applied,
        )

    # ─────────────────────────────────────────
    # INTERNAL HELPERS
    # ─────────────────────────────────────────

    def _apply_hints(
        self,
        hints: List[str],
        epochs: int,
        lr: float,
        n_samples: int,
        arch_name: str,
        applied: List[str],
    ) -> Tuple[int, float, int, str]:
        """Parse hint strings and adjust hyperparams."""
        for hint in hints:
            h = hint.lower()

            if "increase depth" in h or "more layers" in h:
                # Parse current layers and increment
                parts = arch_name.split("L-")
                if len(parts) == 2:
                    try:
                        current = int(parts[0].split("-")[-1])
                        arch_name = arch_name.replace(
                            f"{current}L", f"{current + 2}L"
                        )
                        applied.append(f"Depth: {current}L → {current+2}L")
                    except ValueError:
                        pass

            if "data size" in h or "more data" in h:
                n_samples = int(n_samples * 1.5)
                applied.append(f"Data: n_samples → {n_samples}")

            if "rlhf" in h or "hallucination" in h:
                lr = lr * 0.5
                applied.append(f"LR halved → {lr:.2e} (anti-hallucination)")

            if "chain-of-thought" in h or "cot loss" in h:
                epochs = epochs + 5
                applied.append(f"Epochs +5 → {epochs} (CoT objective)")

            if "early stopping" in h:
                applied.append("Early stopping enabled")

        return epochs, lr, n_samples, arch_name

    def _build_steps(
        self,
        sequence: List[str],
        task_type: TaskType,
        arch_name: str,
        epochs: int,
        lr: float,
        n_samples: int,
    ) -> List[PlanStep]:
        """Map tool names to full PlanStep objects with kwargs."""

        # Difficulty based on target accuracy implied by task type
        difficulty = "hard" if task_type in [
            TaskType.TRAIN_FROM_SCRATCH,
            TaskType.MULTI_AGENT_DESIGN,
            TaskType.SELF_IMPROVEMENT,
        ] else "medium"

        n_test = max(30, n_samples // 100)

        step_configs: Dict[str, Dict] = {
            "generate_test_cases": {
                "kwargs": {
                    "task_type": task_type.value,
                    "n": n_test,
                    "difficulty": difficulty,
                },
                "reason": f"Build evaluation harness first ({n_test} {difficulty} cases)",
                "required": True,
                "depends_on": [],
            },
            "run_training_job": {
                "kwargs": {
                    "architecture": arch_name,
                    "epochs": epochs,
                    "lr": lr,
                    "data_description": f"synthetic-{task_type.value}",
                },
                "reason": f"Train {arch_name} for {epochs} epochs",
                "required": True,
                "depends_on": ["generate_test_cases"],
            },
            "evaluate_model": {
                "kwargs": {
                    "model_id": arch_name,
                    "test_set": task_type.value,
                },
                "reason": "Measure accuracy, hallucination, reasoning quality",
                "required": True,
                "depends_on": ["run_training_job"],
            },
            "run_safety_check": {
                "kwargs": {
                    "model_id": arch_name,
                    "checks": ["toxicity", "bias", "jailbreak"],
                },
                "reason": "Safety gate before accepting model",
                "required": False,   # agent continues even if safety is borderline
                "depends_on": ["run_training_job"],
            },
        }

        steps = []
        for i, tool_name in enumerate(sequence):
            cfg = step_configs.get(tool_name)
            if cfg:
                steps.append(PlanStep(
                    order=i + 1,
                    tool_name=tool_name,
                    kwargs=cfg["kwargs"],
                    reason=cfg["reason"],
                    required=cfg["required"],
                    depends_on=cfg["depends_on"],
                ))
        return steps

    def _reason(
        self,
        task_type: TaskType,
        hints: List[str],
        applied: List[str],
        epochs: int,
        n_samples: int,
    ) -> str:
        """Generate a short chain-of-thought string."""
        base = f"Task type is '{task_type.value}'."
        if not hints:
            return base + " No history — using defaults."
        return (
            base
            + f" Found {len(hints)} past hint(s). "
            + f"Applied {len(applied)} adjustment(s): "
            + "; ".join(applied[:3])
            + f". Final: epochs={epochs}, n_samples={n_samples}."
        )

    def _score_confidence(
        self,
        task_type: TaskType,
        hints: List[str],
        applied: List[str],
    ) -> float:
        """
        Confidence = how sure the planner is the plan will succeed.
        Higher when we have relevant history.
        """
        base = 0.50
        base += min(0.30, len(applied) * 0.10)   # each hint = +10%
        if task_type in [TaskType.EVALUATION]:
            base += 0.15                           # simpler tasks = more confident
        return min(1.0, base)

    # ─────────────────────────────────────────
    # STATS
    # ─────────────────────────────────────────

    def stats(self) -> str:
        return f"Planner: {self._plan_count} plans built so far."
