from __future__ import annotations

import asyncio
import time
import math
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

from .meta_agent import MetaLearningLoop
from ..training.plane import (
    TrainingPlan, TargetMetrics, ArchitectureConfig,
    DataConfig, TrainingConfig, EvaluationMetrics,
)
from ..tools.registry import ToolRegistry, build_default_registry
from ..tools.file_ops import FileOps
from ..experience.episodes import (
    ExperienceEpisode, TaskType, EpisodeStatus,
    EvaluationResult, FailureMode,
)
from ..experience.processor import ReflectionProcessor, UnifiedReport
from ..genome.dna import CognitiveDNA
from ..genome.evolution import EvolutionEngine

try:
    from .planner import Planner
    _HAS_PLANNER = True
except ImportError:
    _HAS_PLANNER = False

try:
    from .intuition import IntuitionEngine
    _HAS_INTUITION = True
except ImportError:
    _HAS_INTUITION = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class Observation:
    task_type: TaskType
    confidence: float
    data_file_hint: Optional[str]
    expected_output: str
    language: str
    entities: List[str]


# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────

class AgentConfig:
    def __init__(
        self,
        budget_usd: float               = 1.0,
        auto_save: bool                 = True,
        verbose: bool                   = True,
        max_plan_retries: int           = 2,
        max_tool_retries: int           = 2,
        intuition_threshold: float      = 0.3,
        uncertainty_trial_threshold: float = 0.4,
        low_quality_threshold: float    = 0.5,
    ):
        self.budget_usd                  = budget_usd
        self.auto_save                   = auto_save
        self.verbose                     = verbose
        self.max_plan_retries            = max_plan_retries
        self.max_tool_retries            = max_tool_retries
        self.intuition_threshold         = intuition_threshold
        self.uncertainty_trial_threshold = uncertainty_trial_threshold
        self.low_quality_threshold       = low_quality_threshold


# ─────────────────────────────────────────────
# PrimeAgent
# ─────────────────────────────────────────────

class PrimeAgent:
    """
    Master orchestrator — plans, executes, adapts, and evolves.

    Loop:
      observe → plan → verify → act → reflect → evolve → store
    """

    def __init__(
            self,
            config: Optional[AgentConfig] = None,
            store_dir: Path = Path("experiments/episodes"),
            dna_path: Path = Path("config/agent_dna.yaml"),
    ):
        self.config = config or AgentConfig()
        self.store_dir = store_dir
        self.dna_path = dna_path
        self.store_dir.mkdir(parents=True, exist_ok=True)

        self.history: List[ExperienceEpisode] = []
        self._load_history()

        self.registry = build_default_registry()
        self.file_ops = FileOps()
        self.reflector = ReflectionProcessor()
        self.dna = self._load_dna()

        # LLM initialization – always set self.llm
        self.llm = None
        try:
            from llm.provider import ModelProvider
            self.llm = ModelProvider.from_env()
            self._log("LLM", str(self.llm))
        except Exception as e:
            logger.warning(f"LLM not available: {e}")

        self.evolution = EvolutionEngine(dna_path=dna_path, llm=self.llm)

        if _HAS_PLANNER:
            self.planner = Planner(self.registry)
        else:
            self.planner = None

        if _HAS_INTUITION:
            self.intuition = IntuitionEngine(self.history, self.dna)
        else:
            self.intuition = None

        self.current_episode: Optional[ExperienceEpisode] = None
        self.current_plan: Optional[TrainingPlan] = None
    
    # Init helpers
    # ─────────────────────────────────────────

    def _load_history(self) -> None:
        for f in sorted(self.store_dir.glob("*.json")):
            if f.name.startswith("_"):
                continue
            try:
                ep = ExperienceEpisode.load(str(f))
                self.history.append(ep)
            except Exception as e:
                logger.warning(f"Failed to load {f}: {e}")
        self.history.sort(key=lambda e: e.started_at)

    def _load_dna(self) -> CognitiveDNA:
        if self.dna_path.exists():
            try:
                return CognitiveDNA.load(self.dna_path)
            except Exception:
                pass
        return CognitiveDNA(dna_id="initial_dna", genes=[])

    def _save_dna(self) -> None:
        try:
            self.dna.save(self.dna_path)
        except Exception:
            pass

    # ─────────────────────────────────────────
    # Chain-of-thought logging
    # ─────────────────────────────────────────

    def _log(self, step: str, details: str = "") -> None:
        if self.config.verbose:
            logger.info(f"[{step}] {details}" if details else step)

    # ─────────────────────────────────────────
    # Observe (enhanced with LLM)
    # ─────────────────────────────────────────

    def observe(self, message: str) -> TaskType:
        # Try LLM first (new enhanced version)
        if hasattr(self, 'llm') and self.llm:
            try:
                loop = asyncio.new_event_loop()
                obs = loop.run_until_complete(self._observe_with_llm(message))
                loop.close()
                if obs:
                    self._log("LLM observe", f"type={obs.task_type.value}, conf={obs.confidence}")
                    # optionally store obs in self.last_observation for later use
                    return obs.task_type
            except Exception as e:
                logger.warning(f"LLM observe failed, falling back to rules: {e}")

        # Fallback to rule‑based (original logic)
        t = message.lower()
        if "train" in t or "from scratch" in t:
            return TaskType.TRAIN_FROM_SCRATCH
        if "fine" in t or "finetune" in t:
            return TaskType.FINE_TUNE
        if "evaluat" in t or "eval" in t:
            return TaskType.EVALUATION
        if "architect" in t or "design" in t:
            return TaskType.ARCHITECTURE_DESIGN
        if "improve" in t or "self" in t:
            return TaskType.SELF_IMPROVEMENT
        return TaskType.UNKNOWN

    # ─────────────────────────────────────────
    # LLM‑powered structured observation
    # ─────────────────────────────────────────
    async def _observe_with_llm(self, user_message: str) -> Optional[Observation]:
        if not self.llm:
            return None
        prompt = f"""Extract structured info from: {user_message}
    Return ONLY valid JSON with these keys:
    {{
        "task_type": "train_from_scratch",
        "confidence": 0.9,
        "data_file_hint": null,
        "expected_output": "trained model",
        "language": "en",
        "entities": ["transformer"]
    }}
    Possible task_type values: train_from_scratch, fine_tune, evaluate, reasoning, general, data_analysis.
    """
        try:
            resp = self.llm.think(prompt)
            if not resp.success:
                return None
            import json
            data = json.loads(resp.content.strip())
            mapping = {
                "train_from_scratch": TaskType.TRAIN_FROM_SCRATCH,
                "fine_tune": TaskType.FINE_TUNE,
                "evaluate": TaskType.EVALUATION,
                "reasoning": TaskType.ARCHITECTURE_DESIGN,
                "general": TaskType.UNKNOWN,
                "data_analysis": TaskType.UNKNOWN,
            }
            return Observation(
                task_type=mapping.get(data.get("task_type"), TaskType.UNKNOWN),
                confidence=data.get("confidence", 0.5),
                data_file_hint=data.get("data_file_hint"),
                expected_output=data.get("expected_output", "model"),
                language=data.get("language", "en"),
                entities=data.get("entities", [])
            )
        except Exception as e:
            logger.warning(f"LLM observe failed: {e}")
            return None
    # ─────────────────────────────────────────
    # Plan
    # ─────────────────────────────────────────

    def _build_plan(self, task: str, task_type: TaskType) -> List[Dict]:
        hints = [ep.next_improvement for ep in self.history[-5:] if ep.next_improvement]

        if hasattr(self, 'llm') and self.llm:
            try:
                hints_text = "\n".join(hints[:3]) if hints else "No history"
                llm_plan = self.llm.plan(task, context=f"Task type: {task_type.value}\n{hints_text}")
                if llm_plan and "error" not in llm_plan and "raw" not in llm_plan:
                    self._log("LLM plan", str(list(llm_plan.keys())))
            except Exception as e:
                self._log("LLM plan failed", str(e))
                pass

        return [
            {"tool": "generate_test_cases", "params": {"task_type": task_type.value, "n": 50}},
            {"tool": "run_training_job", "params": {"epochs": 10, "lr": 3e-4}},
            {"tool": "evaluate_model", "params": {}},
            {"tool": "run_safety_check", "params": {}},
        ]

    # ─────────────────────────────────────────
    # Verify
    # ─────────────────────────────────────────

    def _verify_plan(self, plan: List[Dict], task_type: TaskType) -> bool:
        if self.intuition and _HAS_INTUITION:
            try:
                result = self.intuition.evaluate_plan(
                    plan_steps=plan,
                    task_type=task_type,
                    budget_usd=self.config.budget_usd,
                )
                return result.recommended
            except Exception:
                pass
        return True

    # ─────────────────────────────────────────
    # Act
    # ─────────────────────────────────────────

    def _act(
        self,
        plan:    List[Dict],
        episode: ExperienceEpisode,
    ) -> Tuple[bool, EvaluationResult]:

        step_results = []
        start_time   = time.time()
        total_cost   = 0.0
        last_result  = None

        for step in plan:
            tool_name = step.get("tool", "")
            params    = step.get("params", {})
            success   = False
            duration  = 0.0

            for attempt in range(self.config.max_tool_retries + 1):
                try:
                    t0          = time.time()
                    tool_result = self.registry.run(tool_name, episode=episode, **params)
                    duration    = time.time() - t0
                    success     = tool_result.success
                    last_result = tool_result

                    if not success and attempt < self.config.max_tool_retries:
                        err = tool_result.error_message.lower()
                        if "timeout" in err:
                            params["timeout"] = params.get("timeout", 30) * 2
                        elif "memory" in err and "batch_size" in params:
                            params["batch_size"] = max(4, params["batch_size"] // 2)
                    else:
                        break
                except Exception as exc:
                    duration    = time.time() - t0 if 't0' in locals() else 0.0
                    last_result = None
                    success     = False
                    logger.error(f"Tool {tool_name} exception: {exc}")
                    break

            step_results.append(success)
            total_cost += 0.01 if tool_name == "run_training_job" else 0.001

            if not success:
                logger.error(f"Tool {tool_name} failed — stopping pipeline")
                break

        all_success    = all(step_results)
        duration_total = time.time() - start_time

        # parse metrics from last tool output
        accuracy = hallucination = reasoning = safety = None
        if last_result and last_result.output:
            import re
            out = str(last_result.output)
            for key, pat in [
                ("accuracy",     r"acc(?:uracy)?=([0-9.]+)"),
                ("hallucination",r"hallu(?:cination)?=([0-9.]+)"),
                ("reasoning",    r"r(?:easoning_)?q(?:uality)?=([0-9.]+)"),
                ("safety",       r"safety=([0-9.]+)"),
            ]:
                m = re.search(pat, out)
                if m:
                    val = float(m.group(1))
                    if key == "accuracy":     accuracy     = val
                    if key == "hallucination":hallucination = val
                    if key == "reasoning":   reasoning    = val
                    if key == "safety":      safety       = val

        evaluation = EvaluationResult(
            accuracy=accuracy,
            reasoning_quality=reasoning,
            hallucination_rate=hallucination,
            safety_score=safety,
            plan_efficiency=sum(step_results) / max(len(plan), 1),
            tool_usage_correct=all_success,
            learned_from_history=len(self.history) > 0,
            duration_seconds=duration_total,
            token_cost_estimate=total_cost,
        )
        return all_success, evaluation

    # ─────────────────────────────────────────
    # Trial burn
    # ─────────────────────────────────────────

    def _trial_burn(self, plan: List[Dict], episode: ExperienceEpisode) -> bool:
        trial_steps = [
            {"tool": "generate_test_cases", "params": {"n": 10}},
            {"tool": "run_training_job",    "params": {"epochs": 1}},
        ]
        _, ev = self._act(trial_steps, episode)
        return (ev.accuracy or 0) > 0.3

    # ─────────────────────────────────────────
    # run() — main public method
    # ─────────────────────────────────────────

    def run(self, user_message: str) -> Dict[str, Any]:
        self._log("Processing", user_message)

        # 1. Observe
        task_type = self.observe(user_message)
        self._log("Task type", task_type.value)

        # 2. Plan with retries
        plan     = None
        accepted = False
        for attempt in range(self.config.max_plan_retries + 1):
            plan     = self._build_plan(user_message, task_type)
            accepted = self._verify_plan(plan, task_type)
            if accepted:
                break
            self._log(f"Plan rejected — retry {attempt + 1}")

        if not accepted:
            self._log("Proceeding with low-confidence plan")

        # 3. Trial burn if uncertainty high
        hints    = [ep.next_improvement for ep in self.history[-3:] if ep.next_improvement]
        training_plan = TrainingPlan.design(
            task=user_message,
            task_type=task_type,
            history=hints,
        )
        self.current_plan = training_plan

        if training_plan.uncertainty_score > self.config.uncertainty_trial_threshold:
            self._log("High uncertainty — running trial burn")

        # 4. Create episode and act
        episode = ExperienceEpisode.start(
            user_message,
            task_type,
            uncertainty=training_plan.uncertainty_score,
        )
        self.current_episode = episode

        success, evaluation = self._act(plan, episode)

        # 5. Finish episode
        status = EpisodeStatus.SUCCESS if success else EpisodeStatus.PARTIAL
        fm     = FailureMode.TOOL_ERROR if not success else FailureMode.NONE
        episode.finish(
            status,
            evaluation,
            next_improvement=training_plan.targets.min_accuracy.__class__.__name__,
            failure_mode=fm,
        )

        # override next_improvement with something useful
        parts = []
        if evaluation.accuracy is not None and evaluation.accuracy < training_plan.targets.min_accuracy:
            parts.append("Increase model depth or data size")
        if evaluation.hallucination_rate is not None and evaluation.hallucination_rate > training_plan.targets.max_hallucination:
            parts.append("Add RLHF or contrastive loss")
        episode.next_improvement = " | ".join(parts) if parts else "All targets met"

        # 6. Reflect
        try:
            try:
                self.reflector.llm = self.llm
                loop = asyncio.new_event_loop()
                report = loop.run_until_complete(
                    self.reflector.process_async(episode, self.history)
                )
                loop.close()
            except Exception as e:
                logger.warning(f"Async reflect failed: {e}")
                report = self.reflector.process(episode, self.history)
            if self.config.verbose:
                print(report.full_report())
        except Exception as e:
            logger.warning(f"Reflection failed: {e}")
            report = None

        # 7. Evolve DNA
        try:
            if report:
                self.evolution.evolve(report, self.dna.to_dict())
                self._save_dna()
        except Exception as e:
            logger.warning(f"Evolution failed: {e}")

        # 8. Store
        self.history.append(episode)
        if self.config.auto_save:
            try:
                episode.save(self.store_dir)
            except Exception as e:
                logger.warning(f"Save failed: {e}")

        return {
            "plan_id":    episode.episode_id[:8],
            "steps":      [{"tool": s.get("tool"), "success": True} for s in plan],
            "goals_met":  success,
            "status":     status.value,
            "reflection": episode.next_improvement,
            "next_step":  episode.next_improvement,
        }

    # ─────────────────────────────────────────
    # run_on_file() — file-based training (future: needs torch)
    # ─────────────────────────────────────────

    def run_on_file(self, file_path: Path, task_description: str = "") -> Dict[str, Any]:
        self._log("run_on_file", f"File: {file_path.name}")

        # 1. Profile the file using universal trainer (lightweight analysis)
        from training.universal_trainer import UniversalAutonomousWrapper
        wrapper = UniversalAutonomousWrapper(self.file_ops)

        try:
            # Run full training (simulation by default, can add real flag later)
            report = wrapper.run(file_path, epochs=10, batch_size=8)
        except Exception as e:
            self._log("File processing failed", str(e))
            return {"status": "error", "reason": str(e)}

        # 2. Create an episode from the report
        task_desc = task_description or f"Train on {file_path.name}"
        task_type = self.observe(task_desc)
        episode = ExperienceEpisode.start(task_desc, task_type)

        # Record the universal trainer as a tool call
        episode.record_tool(
            tool_name="universal_trainer",
            input_summary=str(file_path),
            output_summary=f"acc={report.final_accuracy:.3f}, loss={report.training_loss:.3f}",
            success=report.final_accuracy > 0.5,
            duration=0.0,
        )

        # Build evaluation result from report
        evaluation = EvaluationResult(
            accuracy=report.final_accuracy,
            reasoning_quality=report.final_accuracy,  # placeholder
            hallucination_rate=0.0,  # not measured in simple trainer
            safety_score=0.0,
            plan_efficiency=1.0,
            tool_usage_correct=True,
            learned_from_history=len(self.history) > 0,
            duration_seconds=0.0,
            token_cost_estimate=0.0,
            notes=f"Data profile: {report.data_profile.num_samples} samples, lang={report.data_profile.language}",
        )

        # Finish episode
        status = EpisodeStatus.SUCCESS if report.final_accuracy > 0.5 else EpisodeStatus.PARTIAL
        episode.finish(status, evaluation, next_improvement=report.next_improvement)

        # 3. Reflect and evolve
        unified_report = self.reflector.process(episode, self.history)
        self.evolution.evolve(unified_report, self.dna.to_dict())

        # 4. Store
        if self.config.auto_save:
            episode.save(self.store_dir)
            self.history.append(episode)

        # 5. Return result
        return {
            "status": "success",
            "episode_id": episode.episode_id,
            "data_profile": {
                "num_samples": report.data_profile.num_samples,
                "language": report.data_profile.language,
                "vocab_size": report.data_profile.vocab_size_est,
            },
            "model_config": report.model_config,
            "final_accuracy": report.final_accuracy,
            "gene_hints": report.gene_hints,
            "next_improvement": report.next_improvement,
        }
    # ─────────────────────────────────────────
    # Stats
    # ─────────────────────────────────────────

    def stats(self) -> str:
        from experience.episodes import EpisodeStore
        store = EpisodeStore(self.store_dir)
        s     = store.get_statistics()
        return (
            f"Agent Stats — {s.get('total', 0)} episodes\n"
            f"  Success     : {s.get('successful', 0)} "
            f"({s.get('success_rate', 0) * 100:.0f}%)\n"
            f"  Avg accuracy: {s.get('avg_accuracy', 0):.3f}\n"
            f"  Total cost  : ${s.get('total_cost_usd', 0):.4f}\n"
            f"  DNA version : {getattr(self.dna, 'generation', 0)}"
        )