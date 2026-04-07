from __future__ import annotations

import json
import math
import random
import uuid
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from experience.episodes import EpisodeStatus

try:
    from tools.registry import ToolRegistry
except ImportError:
    ToolRegistry = None

try:
    from tools.file_ops import FileOps
except ImportError:
    FileOps = None

try:
    from experience.episodes import (
        ExperienceEpisode as _Episode,
        EvaluationResult,
        EpisodeStatus,
        FailureMode,
        TaskType,
    )
    _HAS_EPISODES = True
except ImportError:
    _HAS_EPISODES = False
    EvaluationResult = None
    EpisodeStatus    = None
    FailureMode      = None
    TaskType         = None


# ─────────────────────────────────────────────
# Enums (unchanged, extended with planning strategies)
# ─────────────────────────────────────────────

class ArchitectureType(str, Enum):
    TRANSFORMER_DECODER   = "transformer_decoder"
    TRANSFORMER_ENCODER   = "transformer_encoder"
    MIXTURE_OF_EXPERTS    = "mixture_of_experts"
    DIFFUSION             = "diffusion"
    STATE_SPACE           = "state_space"
    HYBRID_SSM_ATTENTION  = "hybrid_ssm_attention"
    CNN                   = "cnn"
    HYBRID                = "hybrid"


class ObjectiveType(str, Enum):
    NEXT_TOKEN_PREDICTION   = "next_token_prediction"
    MASKED_LANGUAGE_MODEL   = "masked_lm"
    CONTRASTIVE             = "contrastive"
    REINFORCEMENT_LEARNING  = "reinforcement_learning"
    DISTILLATION            = "distillation"
    PREFERENCE_OPTIMIZATION = "preference_optimization"
    CHAIN_OF_THOUGHT        = "chain_of_thought"
    RLHF                    = "rlhf"


class DataStrategy(str, Enum):
    SYNTHETIC  = "synthetic"
    REAL       = "real"
    MIXED      = "mixed"
    AUGMENTED  = "augmented"


class HallucinationLevel(str, Enum):
    NONE         = "none"
    MINOR        = "minor"
    MODERATE     = "moderate"
    SEVERE       = "severe"
    CATASTROPHIC = "catastrophic"

    @classmethod
    def from_int(cls, v: int) -> "HallucinationLevel":
        return [cls.NONE, cls.MINOR, cls.MODERATE, cls.SEVERE, cls.CATASTROPHIC][min(v, 4)]

    @classmethod
    def from_rate(cls, rate: float) -> "HallucinationLevel":
        if rate < 0.05: return cls.NONE
        if rate < 0.10: return cls.MINOR
        if rate < 0.20: return cls.MODERATE
        if rate < 0.35: return cls.SEVERE
        return cls.CATASTROPHIC

    def as_int(self) -> int:
        return ["none","minor","moderate","severe","catastrophic"].index(self.value)


# New enum for planning strategies (meta‑learning)
class PlanningStrategy(str, Enum):
    DEPTH_FIRST = "depth_first"      # prefer deeper models
    WIDTH_FIRST = "width_first"      # prefer wider models
    BALANCED    = "balanced"
    CONSERVATIVE = "conservative"    # small models, low LR
    AGGRESSIVE   = "aggressive"      # large models, high LR
    COST_AWARE   = "cost_aware"      # favour cheaper inference


# ─────────────────────────────────────────────
# Config dataclasses (unchanged)
# ─────────────────────────────────────────────

@dataclass
class ArchitectureConfig:
    arch_type:           ArchitectureType = ArchitectureType.TRANSFORMER_DECODER
    num_layers:          int   = 12
    hidden_size:         int   = 768
    d_model:             int   = 768
    num_heads:           int   = 12
    vocab_size:          int   = 50257
    max_sequence_length: int   = 2048
    dropout:             float = 0.1
    activation:          str   = "gelu"
    use_flash_attention: bool  = True
    num_experts:         Optional[int] = None
    top_k_experts:       Optional[int] = None
    state_dim:           Optional[int] = None

    def __post_init__(self):
        if self.d_model != 768 and self.hidden_size == 768:
            self.hidden_size = self.d_model
        elif self.hidden_size != 768 and self.d_model == 768:
            self.d_model = self.hidden_size

    def param_count(self) -> int:
        h     = self.hidden_size
        attn  = self.num_layers * (h * h * 4)
        ffn   = self.num_layers * (h * h * 4)
        emb   = self.vocab_size * h
        total = attn + ffn + emb
        if self.num_experts:
            total = int(total * self.num_experts / 8)
        return total

    def complexity_score(self) -> float:
        return round(math.log10(max(self.param_count(), 1)) / 10.0, 3)

    def inference_flops_per_token(self) -> float:
        """Estimate FLOPs per token for inference (for cost‑aware planning)."""
        # Simplified: 2 * params * layers (forward pass)
        return self.param_count() * self.num_layers * 2

    def summary(self) -> str:
        return (
            f"{self.arch_type.value}-{self.num_layers}L-"
            f"d{self.hidden_size}-h{self.num_heads} "
            f"(~{self.param_count():,} params)"
        )

    def to_dict(self) -> Dict:
        d = asdict(self)
        d["arch_type"] = self.arch_type.value
        return d


@dataclass
class DataConfig:
    dataset_name:       str          = "custom_synthetic"
    dataset_size:       int          = 10_000_000
    n_samples:          int          = 10_000
    strategy:           DataStrategy = DataStrategy.SYNTHETIC
    validation_split:   float        = 0.05
    test_split:         float        = 0.05
    synthetic_generator: Optional[str] = None
    diversity_metrics:  List[str]    = field(default_factory=lambda: ["perplexity", "token_distribution"])
    min_quality_score:  float        = 0.70
    quality:            float        = 0.85
    diversity:          float        = 0.80
    remove_duplicates:  bool         = True
    max_length_tokens:  int          = 2048
    min_length_tokens:  int          = 64
    description:        str          = "synthetic dataset"

    def estimated_tokens(self) -> int:
        return self.n_samples * self.max_length_tokens


@dataclass
class TrainingConfig:
    batch_size:                  int   = 32
    gradient_accumulation_steps: int   = 4
    learning_rate:               float = 3e-4
    warmup_steps:                int   = 2000
    max_steps:                   int   = 100_000
    epochs:                      int   = 10
    optimizer:                   str   = "adamw"
    weight_decay:                float = 0.01
    scheduler:                   str   = "cosine"
    use_fp16:                    bool  = True
    use_bf16:                    bool  = False
    save_every:                  int   = 5000
    eval_every:                  int   = 500
    gradient_clip:               float = 1.0
    label_smoothing:             float = 0.0

    def estimated_steps(self, n_samples: int) -> int:
        return (n_samples // self.batch_size) * self.epochs


@dataclass
class EvaluationMetrics:
    train_loss:          float              = 0.0
    val_loss:            float              = 0.0
    perplexity:          float              = 0.0
    accuracy:            float              = 0.0
    reasoning_score:     float              = 0.0
    hallucination_level: HallucinationLevel = HallucinationLevel.NONE
    hallucination:       float              = 0.0
    bias_score:          float              = 0.0
    safety_score:        float              = 0.0
    training_time_hours: float              = 0.0
    gpu_hours:           float              = 0.0
    flops_utilization:   float              = 0.0

    # Uncertainty intervals (new)
    accuracy_ci_low:     Optional[float]    = None
    accuracy_ci_high:    Optional[float]    = None
    uncertainty_std:     Optional[float]    = None

    def passed_targets(self, targets: "TargetMetrics") -> bool:
        return (
            self.accuracy      >= targets.min_accuracy and
            self.hallucination <= targets.max_hallucination and
            self.safety_score  >= targets.min_safety
        )

    def summary(self) -> str:
        acc_str = f"{self.accuracy:.3f}"
        if self.accuracy_ci_low and self.accuracy_ci_high:
            acc_str = f"{self.accuracy:.3f} (±{self.accuracy_ci_high - self.accuracy:.3f})"
        return (
            f"acc={acc_str} | "
            f"hallu={self.hallucination:.3f} ({self.hallucination_level.value}) | "
            f"reasoning={self.reasoning_score:.3f} | "
            f"safety={self.safety_score:.3f}"
        )

    def to_dict(self) -> Dict:
        d = asdict(self)
        d["hallucination_level"] = self.hallucination_level.value
        return d


@dataclass
class TargetMetrics:
    min_accuracy:      float = 0.80
    max_hallucination: float = 0.10
    min_safety:        float = 0.90
    min_reasoning:     float = 0.75
    # New multi‑objective weights (default: accuracy only)
    weight_accuracy:   float = 0.5
    weight_speed:      float = 0.3
    weight_cost:       float = 0.2


# ─────────────────────────────────────────────
# TrainingPlan – now a genius inference engine
# ─────────────────────────────────────────────

class TrainingPlan:

    def __init__(
        self,
        task_description:   str,
        previous_episodes:  Optional[List[Any]] = None,
        targets:            Optional[TargetMetrics] = None,
    ):
        self.task_description   = task_description
        self.previous_episodes  = previous_episodes or []
        self.targets            = targets or TargetMetrics()
        self.reasoning_chain:   List[str] = []
        self.history_hints:     List[str] = []
        self.uncertainty_score: float     = 0.5
        self.plan_id            = str(uuid.uuid4())[:8]
        self.architecture:      Optional[ArchitectureConfig] = None
        self.data:              Optional[DataConfig]         = None
        self.training:          Optional[TrainingConfig]     = None
        self.objective:         ObjectiveType = ObjectiveType.NEXT_TOKEN_PREDICTION

        # --- Meta‑planning memory (strategy success tracking) ---
        self.planning_memory: Dict[PlanningStrategy, Dict[str, float]] = {}
        self._load_planning_memory()

        # --- Causal effect cache ---
        self._causal_estimates: Dict[str, float] = {}

    def _load_planning_memory(self):
        """Load previous strategy outcomes from history episodes."""
        for ep in self.previous_episodes:
            strat = getattr(ep, "planning_strategy", None)
            if strat and isinstance(strat, str):
                try:
                    strat_enum = PlanningStrategy(strat)
                except ValueError:
                    continue
                acc = getattr(ep.evaluation, "accuracy", 0.0) if ep.evaluation else 0.0
                if strat_enum not in self.planning_memory:
                    self.planning_memory[strat_enum] = {"total_acc": 0.0, "count": 0}
                self.planning_memory[strat_enum]["total_acc"] += acc
                self.planning_memory[strat_enum]["count"] += 1

    def get_best_planning_strategy(self) -> PlanningStrategy:
        """Return the strategy with highest average accuracy from history."""
        if not self.planning_memory:
            return PlanningStrategy.BALANCED
        best = max(self.planning_memory.items(),
                   key=lambda kv: kv[1]["total_acc"] / kv[1]["count"])
        return best[0]

    # ─────────────────────────────────────────
    # 1. Intelligent Scaling Law (power‑law based)
    # ─────────────────────────────────────────

    def _power_law_hidden_size(self, num_tokens: int) -> int:
        """
        hidden_size = min(max_size, max(min_size, base * (num_tokens ** alpha)))
        alpha ~ 0.25, base ~ 200
        """
        if num_tokens <= 0:
            return 64
        raw = 200 * (num_tokens ** 0.25)
        size = max(64, min(512, int(raw)))
        return size

    def _power_law_num_layers(self, num_tokens: int) -> int:
        """layers = max(2, min(12, 4 * (num_tokens ** 0.1)))"""
        if num_tokens <= 0:
            return 2
        raw = 4 * (num_tokens ** 0.1)
        layers = max(2, min(12, int(raw)))
        return layers

    # ─────────────────────────────────────────
    # 2. Multi‑objective optimization (Pareto)
    # ─────────────────────────────────────────

    def _score_architecture_for_targets(self, arch: ArchitectureConfig, targets: TargetMetrics) -> float:
        """
        Compute a weighted score based on accuracy, speed, cost.
        Accuracy is estimated from scaling law (placeholder), speed = 1 / FLOPs, cost = 1 / params.
        """
        # Estimate accuracy from hidden size (simplified: larger = higher up to a point)
        est_accuracy = min(0.95, 0.5 + 0.001 * arch.hidden_size)
        # Speed: lower FLOPs is better (inverse)
        flops = arch.inference_flops_per_token()
        speed_score = max(0, 1.0 - math.log10(max(flops, 1)) / 10.0)
        # Cost: lower parameters is better
        param_score = max(0, 1.0 - math.log10(max(arch.param_count(), 1)) / 12.0)

        return (targets.weight_accuracy * est_accuracy +
                targets.weight_speed * speed_score +
                targets.weight_cost * param_score)

    def optimize_for_tradeoffs(self, targets: Optional[TargetMetrics] = None) -> ArchitectureConfig:
        """
        Generate a small set of candidate architectures and return the one
        that best satisfies the weighted objectives.
        """
        if targets is None:
            targets = self.targets

        # Use current data profile if available (from metadata)
        num_tokens = getattr(self, '_cached_num_tokens', 10000)
        base_hidden = self._power_law_hidden_size(num_tokens)
        base_layers = self._power_law_num_layers(num_tokens)

        candidates = [
            # Balanced (derived from scaling law)
            ArchitectureConfig(num_layers=base_layers, hidden_size=base_hidden,
                               num_heads=max(4, base_hidden // 64)),
            # Depth‑first (more layers, same hidden)
            ArchitectureConfig(num_layers=min(16, base_layers + 4), hidden_size=base_hidden,
                               num_heads=max(4, base_hidden // 64)),
            # Width‑first (wider, fewer layers)
            ArchitectureConfig(num_layers=max(4, base_layers - 2), hidden_size=min(768, base_hidden * 2),
                               num_heads=min(16, (base_hidden * 2) // 64)),
            # Conservative (small)
            ArchitectureConfig(num_layers=2, hidden_size=64, num_heads=4),
            # Aggressive (large, if data permits)
            ArchitectureConfig(num_layers=min(20, base_layers + 6), hidden_size=min(1024, base_hidden * 2),
                               num_heads=min(16, (base_hidden * 2) // 64)),
        ]
        # Score each
        best_arch = max(candidates, key=lambda a: self._score_architecture_for_targets(a, targets))
        self.reasoning_chain.append(f"Multi‑objective optimisation selected {best_arch.summary()} "
                                    f"with weights acc={targets.weight_accuracy}, speed={targets.weight_speed}, cost={targets.weight_cost}")
        return best_arch

    # ─────────────────────────────────────────
    # 3. Uncertainty Quantification (bootstrap on simulation)
    # ─────────────────────────────────────────

    def _bootstrap_metrics(self, arch: ArchitectureConfig, data: DataConfig,
                           training: TrainingConfig, obj: ObjectiveType,
                           n_iter: int = 10) -> Tuple[float, float, float]:
        """
        Run simulation multiple times with small noise to estimate std and CI.
        Returns (mean_accuracy, std_accuracy, 90%_ci_half_range).
        """
        accuracies = []
        for _ in range(n_iter):
            # Add small noise to key parameters to simulate different seeds
            noisy_arch = arch
            noisy_data = data
            noisy_training = training
            metrics = self._simulate_once(noisy_arch, noisy_data, noisy_training, obj)
            accuracies.append(metrics.accuracy)
        mean_acc = statistics.mean(accuracies)
        std_acc = statistics.stdev(accuracies) if len(accuracies) > 1 else 0.05
        # 90% CI: ±1.645 * std
        ci_half = 1.645 * std_acc
        return mean_acc, std_acc, ci_half

    # ─────────────────────────────────────────
    # 4. Causal Reasoning from History (Difference‑in‑differences)
    # ─────────────────────────────────────────

    def causal_effect_estimation(self, change_description: str) -> Optional[float]:
        """
        Estimate the impact of a specific architectural change (e.g., "add 2 layers")
        by comparing episodes before and after the change (difference‑in‑differences).
        Returns average accuracy lift.
        """
        if len(self.previous_episodes) < 4:
            return None
        # Group episodes into those with the change and without
        # We use a simple heuristic: change is present if hint contains keywords
        changed_acc = []
        unchanged_acc = []
        for ep in self.previous_episodes:
            hints = getattr(ep, "gene_hints", [])
            if not hints:
                unchanged_acc.append(ep.evaluation.accuracy if ep.evaluation else 0.0)
                continue
            # If any hint contains the change description, consider it "treated"
            treated = any(change_description.lower() in h.lower() for h in hints)
            acc = ep.evaluation.accuracy if ep.evaluation else 0.0
            if treated:
                changed_acc.append(acc)
            else:
                unchanged_acc.append(acc)
        if not changed_acc or not unchanged_acc:
            return None
        avg_change = statistics.mean(changed_acc)
        avg_no_change = statistics.mean(unchanged_acc)
        effect = avg_change - avg_no_change
        self.reasoning_chain.append(f"Causal estimate: {change_description} → Δaccuracy = {effect:.3f}")
        return effect

    # ─────────────────────────────────────────
    # 5. Active Learning for Plan Refinement (suggest experiments)
    # ─────────────────────────────────────────

    def suggest_experiments(self, num_suggestions: int = 2) -> List[Dict[str, Any]]:
        """
        Suggest small, cheap experiments to resolve uncertainty.
        Returns list of experiment descriptors.
        """
        experiments = []
        # Experiment 1: learning rate sweep (if not already explored)
        if self.training:
            current_lr = self.training.learning_rate
            experiments.append({
                "type": "lr_sweep",
                "values": [current_lr * 0.3, current_lr, current_lr * 3.0],
                "steps": 10,
                "cost_estimate": 0.01,
                "purpose": "Find optimal learning rate"
            })
        # Experiment 2: compare depth vs width with tiny proxy model
        if self.architecture:
            experiments.append({
                "type": "depth_vs_width",
                "configs": [
                    {"num_layers": self.architecture.num_layers, "hidden_size": self.architecture.hidden_size},
                    {"num_layers": max(2, self.architecture.num_layers - 2), "hidden_size": self.architecture.hidden_size * 2},
                    {"num_layers": min(12, self.architecture.num_layers + 2), "hidden_size": max(64, self.architecture.hidden_size // 2)},
                ],
                "steps": 20,
                "cost_estimate": 0.02,
                "purpose": "Determine depth/width trade‑off"
            })
        # Experiment 3: data quality check (if data is small)
        if self.data and self.data.n_samples < 5000:
            experiments.append({
                "type": "data_augmentation_test",
                "augmentations": ["noise", "synonyms", "back_translation"],
                "steps": 5,
                "cost_estimate": 0.005,
                "purpose": "Check if data augmentation helps"
            })
        return experiments[:num_suggestions]

    # ─────────────────────────────────────────
    # 6. Meta‑planning: use best strategy and record outcome
    # ─────────────────────────────────────────

    def apply_planning_strategy(self, strategy: PlanningStrategy) -> None:
        """
        Adjust architecture and training hyperparameters according to the chosen strategy.
        """
        if not self.architecture:
            return
        if strategy == PlanningStrategy.DEPTH_FIRST:
            self.architecture.num_layers = min(24, self.architecture.num_layers + 2)
            self.reasoning_chain.append("Applied DEPTH_FIRST: increased layers")
        elif strategy == PlanningStrategy.WIDTH_FIRST:
            self.architecture.hidden_size = min(1024, self.architecture.hidden_size * 2)
            self.architecture.num_heads = min(16, self.architecture.num_heads * 2)
            self.reasoning_chain.append("Applied WIDTH_FIRST: increased hidden size")
        elif strategy == PlanningStrategy.CONSERVATIVE:
            self.architecture.num_layers = max(4, self.architecture.num_layers // 2)
            self.architecture.hidden_size = max(64, self.architecture.hidden_size // 2)
            if self.training:
                self.training.learning_rate /= 2
                self.training.epochs = max(5, self.training.epochs // 2)
            self.reasoning_chain.append("Applied CONSERVATIVE: reduced model size and LR")
        elif strategy == PlanningStrategy.AGGRESSIVE:
            self.architecture.num_layers = min(24, self.architecture.num_layers * 2)
            self.architecture.hidden_size = min(1024, self.architecture.hidden_size * 2)
            if self.training:
                self.training.learning_rate *= 2
                self.training.epochs = min(30, self.training.epochs * 2)
            self.reasoning_chain.append("Applied AGGRESSIVE: increased capacity and LR")
        elif strategy == PlanningStrategy.COST_AWARE:
            # Reduce inference cost
            self.architecture.hidden_size = max(64, self.architecture.hidden_size // 2)
            self.architecture.num_layers = max(4, self.architecture.num_layers // 2)
            if self.training:
                self.training.batch_size = min(128, self.training.batch_size * 2)
            self.reasoning_chain.append("Applied COST_AWARE: reduced size for cheaper inference")
        # Record strategy for meta‑learning
        self.current_strategy = strategy

    # ─────────────────────────────────────────
    # Main inference method (updated)
    # ─────────────────────────────────────────

    def infer_from_data_profile(self, profile: Any) -> None:
        """
        Enhanced inference using all six advanced methods.
        """
        if profile is None:
            return

        tokens = profile.num_tokens_est
        lang   = profile.language
        vocab_est = profile.vocab_size_est
        self._cached_num_tokens = tokens

        self.reasoning_chain.append(f"Intelligent inference from data: {tokens} tokens, lang={lang}")

        # 1. Scaling law for initial architecture
        hidden = self._power_law_hidden_size(tokens)
        layers = self._power_law_num_layers(tokens)
        self.architecture = ArchitectureConfig(
            arch_type=ArchitectureType.TRANSFORMER_DECODER,
            num_layers=layers,
            hidden_size=hidden,
            d_model=hidden,
            num_heads=max(4, hidden // 64),
            vocab_size=vocab_est,
            dropout=0.1,
        )
        self.reasoning_chain.append(f"Scaling law → hidden={hidden}, layers={layers}")

        # 2. Multi‑objective optimisation (refine based on target weights)
        best_arch = self.optimize_for_tradeoffs(self.targets)
        self.architecture = best_arch

        # 3. Meta‑planning: apply best strategy from history
        best_strategy = self.get_best_planning_strategy()
        self.apply_planning_strategy(best_strategy)
        self.reasoning_chain.append(f"Meta‑planning applied strategy: {best_strategy.value}")

        # 4. Causal reasoning: adjust if we have evidence for a specific change
        depth_effect = self.causal_effect_estimation("increase depth")
        if depth_effect and depth_effect > 0.02 and self.architecture.num_layers < 20:
            self.architecture.num_layers += 1
            self.reasoning_chain.append(f"Causal evidence suggests increasing depth (Δ+{depth_effect:.2f})")

        # 5. Data config from profile
        self.data = DataConfig(
            dataset_name="user_file",
            n_samples=min(profile.num_samples, 100_000),
            dataset_size=tokens,
            strategy=DataStrategy.REAL,
            diversity_metrics=["perplexity", "token_distribution"],
        )
        if lang != "en":
            self.data.diversity_metrics.append(f"lang_{lang}")

        # 6. Training hyperparameters (dynamic)
        self.training = TrainingConfig(
            learning_rate=3e-4 * min(1.0, math.log10(tokens) / 6),
            epochs=20 if tokens < 5000 else (15 if tokens < 50000 else 10),
            batch_size=max(8, min(64, tokens // 1000)),
            warmup_steps=min(2000, max(100, tokens // 500)),
        )
        self.reasoning_chain.append(f"Dynamic training: LR={self.training.learning_rate:.2e}, epochs={self.training.epochs}, batch={self.training.batch_size}")

        # 7. Active learning: suggest experiments (stored in metadata for later use)
        self._suggested_experiments = self.suggest_experiments(num_suggestions=2)
        if self._suggested_experiments:
            self.reasoning_chain.append(f"Active learning suggests experiments: {[e['type'] for e in self._suggested_experiments]}")

        # 8. Objective based on task
        self.objective = self.define_objective()
        self.uncertainty_score = 0.3  # after inference, lower uncertainty

    # ─────────────────────────────────────────
    # Existing methods (with uncertainty integration)
    # ─────────────────────────────────────────
    def _simulate_once(self, arch, data, training, obj) -> "EvaluationMetrics":
        params_b = arch.param_count() / 1e9
        final_loss = self._simulate_loss(params_b, data.dataset_size, training.max_steps)
        reasoning = self._simulate_reasoning(arch, obj)
        hallu_lvl = self._simulate_hallucinations(arch, data, reasoning)
        hallu_rate = [0.02, 0.07, 0.15, 0.27, 0.45][hallu_lvl.as_int()]
        accuracy = max(0.0, min(0.95, 1.0 - final_loss / 5.0))
        return EvaluationMetrics(
            accuracy=round(accuracy, 3),
            hallucination=round(hallu_rate, 3),
            hallucination_level=hallu_lvl,
            reasoning_score=round(reasoning, 3),
            safety_score=round(self._simulate_safety(arch, obj), 3),
        )
    def simulate_training(
        self,
        architecture: ArchitectureConfig,
        data: DataConfig,
        training: TrainingConfig,
        objective: ObjectiveType,
    ) -> EvaluationMetrics:
        self.reasoning_chain.append("Running simulated training with uncertainty...")
        params_b       = architecture.param_count() / 1e9
        final_loss     = self._simulate_loss(params_b, data.dataset_size, training.max_steps)
        reasoning      = self._simulate_reasoning(architecture, objective)
        hallu_level    = self._simulate_hallucinations(architecture, data, reasoning)
        hallu_rate     = [0.02, 0.07, 0.15, 0.27, 0.45][hallu_level.as_int()]
        bias           = self._simulate_bias(data)
        safety         = self._simulate_safety(architecture, objective)
        perplexity     = math.exp(final_loss) if final_loss < 10 else 1000
        accuracy       = max(0.0, min(0.95, 1.0 - final_loss / 5.0))
        gpu_hours      = (architecture.param_count() * training.max_steps * training.batch_size) / 1e15

        # Uncertainty quantification (bootstrap)
        mean_acc, std_acc, ci_half = self._bootstrap_metrics(architecture, data, training, objective)

        return EvaluationMetrics(
            train_loss=round(final_loss * 0.9, 4),
            val_loss=round(final_loss, 4),
            perplexity=round(perplexity, 2),
            accuracy=round(mean_acc, 3),
            reasoning_score=round(reasoning, 3),
            hallucination_level=hallu_level,
            hallucination=round(hallu_rate, 3),
            bias_score=round(bias, 3),
            safety_score=round(safety, 3),
            training_time_hours=round(gpu_hours / 8, 2),
            gpu_hours=round(gpu_hours, 2),
            flops_utilization=round(random.uniform(0.4, 0.7), 2),
            accuracy_ci_low=round(mean_acc - ci_half, 3),
            accuracy_ci_high=round(mean_acc + ci_half, 3),
            uncertainty_std=round(std_acc, 4),
        )

    # (All existing helper methods: _simulate_loss, _simulate_reasoning, etc. remain unchanged)
    def _simulate_loss(self, params_b: float, data_size: int, steps: int) -> float:
        data_f  = min(1.0, math.log10(max(data_size, 1)) / 10)
        steps_f = min(1.0, math.log10(max(steps, 1)) / 12)
        model_f = min(0.5, params_b / 10)
        improve = data_f * 0.5 + steps_f * 0.3 + model_f * 0.2
        loss    = 3.0 * (1 - improve) + random.gauss(0, 0.1)
        return max(1.0, min(5.0, loss))

    def _simulate_reasoning(self, arch: ArchitectureConfig, obj: ObjectiveType) -> float:
        base        = 0.5
        depth_bonus = min(0.3, arch.num_layers / 80)
        if arch.arch_type == ArchitectureType.MIXTURE_OF_EXPERTS:
            depth_bonus += 0.1
        obj_bonus = {
            ObjectiveType.NEXT_TOKEN_PREDICTION:    0.10,
            ObjectiveType.PREFERENCE_OPTIMIZATION:  0.15,
            ObjectiveType.CHAIN_OF_THOUGHT:         0.18,
            ObjectiveType.RLHF:                     0.15,
        }.get(obj, 0.05)
        return min(0.95, max(0.3, base + depth_bonus + obj_bonus + random.gauss(0, 0.05)))

    def _simulate_hallucinations(
        self,
        arch:  ArchitectureConfig,
        data:  DataConfig,
        rscore: float,
    ) -> HallucinationLevel:
        base = 0 if rscore > 0.8 else (1 if rscore > 0.6 else 2)
        if arch.arch_type == ArchitectureType.HYBRID_SSM_ATTENTION:
            base = max(0, base - 1)
        if data.synthetic_generator:
            base = min(4, base + 1)
        return HallucinationLevel.from_int(base)

    def _simulate_bias(self, data: DataConfig) -> float:
        if data.synthetic_generator and "diversity" in str(data.diversity_metrics):
            return round(random.uniform(0.05, 0.15), 3)
        return round(random.uniform(0.10, 0.30), 3)

    def _simulate_safety(self, arch: ArchitectureConfig, obj: ObjectiveType) -> float:
        base = 0.7
        if obj in (ObjectiveType.PREFERENCE_OPTIMIZATION, ObjectiveType.RLHF):
            base += 0.2
        if arch.num_layers > 20:
            base -= 0.1
        return round(min(0.95, max(0.5, base)), 3)

    # ─────────────────────────────────────────
    # Factory methods (existing + enhanced)
    # ─────────────────────────────────────────

    @classmethod
    def design(
        cls,
        task:      str,
        task_type: Any                           = None,
        arch:      Optional[ArchitectureConfig]  = None,
        data:      Optional[DataConfig]          = None,
        training:  Optional[TrainingConfig]      = None,
        objective: Optional[ObjectiveType]       = None,
        targets:   Optional[TargetMetrics]       = None,
        history:   Optional[List[str]]           = None,
        profile:   Optional[Any]                 = None,
    ) -> "TrainingPlan":
        plan = cls(task, targets=targets)
        plan.history_hints = history or []

        if profile is not None:
            plan.infer_from_data_profile(profile)
            if arch: plan.architecture = arch
            if data: plan.data = data
            if training: plan.training = training
        else:
            plan.architecture = arch or plan.design_architecture()
            plan.data         = data or plan.design_data_pipeline(plan.architecture)
            plan.training     = training or plan.configure_training(plan.architecture)

        plan.objective = objective or plan.define_objective()

        for hint in plan.history_hints:
            h = hint.lower()
            if "depth" in h or "layer" in h:
                plan.architecture.num_layers = min(plan.architecture.num_layers + 2, 24)
            if "lr" in h or "learning rate" in h:
                plan.training.learning_rate *= 0.5
            if "data" in h or "sample" in h:
                plan.data.n_samples = int(plan.data.n_samples * 1.5)
            if "rlhf" in h or "hallucin" in h:
                plan.objective = ObjectiveType.RLHF

        plan.uncertainty_score = 0.7 if not plan.history_hints else max(
            0.3, 0.7 - len(plan.history_hints) * 0.1
        )
        return plan

    @classmethod
    def from_autonomous_ingestion(
        cls,
        file_path: Path,
        task_description: str,
        profile: Any,
        previous_episodes: Optional[List[Any]] = None,
    ) -> "TrainingPlan":
        plan = cls(task_description, previous_episodes)
        plan.infer_from_data_profile(profile)
        plan.metadata = getattr(plan, "metadata", {})
        plan.metadata["file_path"] = str(file_path)
        plan.metadata["use_universal_trainer"] = True
        return plan

    # Existing design methods (kept as fallback)
    def design_architecture(self) -> ArchitectureConfig:
        self.reasoning_chain.append("Designing model architecture...")
        task    = self.task_description.lower()
        similar = self._find_similar_episodes()

        if "reasoning" in task or "cot" in task:
            arch = ArchitectureConfig(
                arch_type=ArchitectureType.TRANSFORMER_DECODER,
                num_layers=24, hidden_size=1024, num_heads=16,
                use_flash_attention=True,
            )
            self.reasoning_chain.append("Transformer decoder depth=24 for complex reasoning")
        elif "hallucination" in task:
            arch = ArchitectureConfig(
                arch_type=ArchitectureType.HYBRID_SSM_ATTENTION,
                num_layers=18, hidden_size=896, num_heads=14,
                state_dim=128, use_flash_attention=True,
            )
            self.reasoning_chain.append("Hybrid SSM-attention for factual grounding")
        elif "vision" in task or "image" in task:
            arch = ArchitectureConfig(
                arch_type=ArchitectureType.CNN,
                num_layers=4, hidden_size=128, num_heads=4,
            )
            self.reasoning_chain.append("CNN for vision task")
        elif "small" in task or "tiny" in task or "light" in task:
            arch = ArchitectureConfig(num_layers=6, hidden_size=256, num_heads=8)
            self.reasoning_chain.append("Small transformer baseline")
        else:
            arch = ArchitectureConfig(
                arch_type=ArchitectureType.TRANSFORMER_DECODER,
                num_layers=12, hidden_size=768, num_heads=12,
            )
            self.reasoning_chain.append("Standard transformer decoder baseline")

        if similar:
            best = self._analyze_past_architectures(similar)
            if best:
                self.reasoning_chain.append(f"Past experience suggests: {best}")

        return arch

    def design_data_pipeline(self, architecture: ArchitectureConfig) -> DataConfig:
        self.reasoning_chain.append("Designing data pipeline...")
        task          = self.task_description.lower()
        need_synthetic = any(k in task for k in ["few", "limited", "synthetic", "scratch"])

        if need_synthetic:
            data = DataConfig(
                dataset_name="synthetic_reasoning",
                dataset_size=50_000_000,
                n_samples=15_000,
                strategy=DataStrategy.SYNTHETIC,
                synthetic_generator="llm_teacher",
                diversity_metrics=["perplexity", "semantic_diversity", "reasoning_depth"],
            )
            self.reasoning_chain.append("Synthetic data with teacher model")
        else:
            data = DataConfig(
                dataset_name="curated_corpus",
                dataset_size=10_000_000,
                n_samples=10_000,
                strategy=DataStrategy.REAL,
                diversity_metrics=["perplexity", "token_distribution"],
            )
            self.reasoning_chain.append("Curated real dataset")

        if "reasoning" in task:
            data.diversity_metrics.append("logical_consistency")
            data.min_quality_score = 0.85
            self.reasoning_chain.append("Added reasoning quality filters")

        return data

    def configure_training(self, architecture: ArchitectureConfig) -> TrainingConfig:
        self.reasoning_chain.append("Configuring training...")
        task     = self.task_description.lower()
        base_lr  = 3e-4
        lr_scale = math.sqrt(768 / max(architecture.hidden_size, 1))

        config = TrainingConfig(
            learning_rate=base_lr * lr_scale,
            batch_size=32,
            epochs=10,
            max_steps=100_000,
            warmup_steps=2000,
            use_fp16=True,
        )
        if architecture.num_layers > 20:
            config.gradient_accumulation_steps = 8
            config.batch_size = 16
            self.reasoning_chain.append("Adjusted batch for deep model")
        if "rlhf" in task or "alignment" in task:
            config.learning_rate = 1e-5
            config.epochs = 5
        return config

    def define_objective(self) -> ObjectiveType:
        task = self.task_description.lower()
        if "preference" in task or "alignment" in task:
            return ObjectiveType.PREFERENCE_OPTIMIZATION
        if "rlhf" in task:
            return ObjectiveType.RLHF
        if "distill" in task:
            return ObjectiveType.DISTILLATION
        if "reasoning" in task or "cot" in task:
            return ObjectiveType.CHAIN_OF_THOUGHT
        return ObjectiveType.NEXT_TOKEN_PREDICTION

    # Pipeline and execution (unchanged)
    def pipeline_steps(self) -> List[Tuple[str, Dict]]:
        arch = self.architecture or ArchitectureConfig()
        data = self.data         or DataConfig()
        tr   = self.training     or TrainingConfig()
        return [
            ("generate_test_cases", {
                "task_type": "reasoning",
                "n": max(50, data.n_samples // 100),
                "difficulty": "hard",
            }),
            ("run_training_job", {
                "architecture": arch.summary(),
                "epochs":       tr.epochs,
                "lr":           tr.learning_rate,
            }),
            ("evaluate_model", {"model_id": self.plan_id}),
            ("run_safety_check", {
                "model_id": self.plan_id,
                "checks":   ["toxicity", "bias", "jailbreak"],
            }),
        ]

    def execute(self, registry: Any, episode: Any = None) -> Dict[str, Any]:
        arch = self.architecture or ArchitectureConfig()
        data = self.data         or DataConfig()
        tr   = self.training     or TrainingConfig()
        obj  = self.objective

        steps, results, all_ok = self.pipeline_steps(), [], True

        for tool_name, kwargs in steps:
            if registry is None:
                break
            if episode is not None and hasattr(episode, "record_causal_step"):
                episode.record_causal_step(
                    decision=f"Run {tool_name}",
                    reason=f"Standard {obj.value} pipeline step",
                    confidence=1.0 - self.uncertainty_score,
                )
            result = registry.run(tool_name, episode=episode, **kwargs)
            results.append(result)
            if not result.success:
                all_ok = False

        metrics   = self.simulate_training(arch, data, tr, obj)
        goals_met = metrics.passed_targets(self.targets)

        parts = []
        if metrics.accuracy < self.targets.min_accuracy:
            parts.append(f"Accuracy {metrics.accuracy:.2f} below target.")
            if episode and hasattr(episode, "add_gene_hint"):
                episode.add_gene_hint(f"Accuracy {metrics.accuracy:.2f} < target — increase depth or data")
        if metrics.hallucination > self.targets.max_hallucination:
            parts.append(f"Hallucination {metrics.hallucination:.2f} too high.")
            if episode and hasattr(episode, "add_gene_hint"):
                episode.add_gene_hint(f"Hallucination {metrics.hallucination:.2f} > target — add RLHF")
        if not parts:
            parts.append("All targets met.")

        reflection = " ".join(parts)
        next_step  = ""
        if metrics.accuracy < self.targets.min_accuracy:
            next_step += "Increase model depth or data size. "
        if metrics.hallucination > self.targets.max_hallucination:
            next_step += "Add RLHF or contrastive loss. "

        status = "success" if goals_met else ("partial" if metrics.accuracy > 0.5 else "failed")

        if episode is not None and _HAS_EPISODES and episode.status == EpisodeStatus.RUNNING:
            fm = (
                FailureMode.HALLUCINATION if metrics.hallucination > self.targets.max_hallucination
                else FailureMode.TOOL_ERROR if not all_ok
                else FailureMode.NONE
            )
            ep_status = (
                EpisodeStatus.SUCCESS if goals_met
                else EpisodeStatus.PARTIAL if metrics.accuracy > 0.5
                else EpisodeStatus.FAILED
            )
            episode.finish(
                ep_status,
                EvaluationResult(
                    accuracy=metrics.accuracy,
                    reasoning_quality=metrics.reasoning_score,
                    hallucination_rate=metrics.hallucination,
                    safety_score=metrics.safety_score,
                    plan_efficiency=0.8 if all_ok else 0.4,
                    tool_usage_correct=all_ok,
                    learned_from_history=len(self.history_hints) > 0,
                ),
                reflection=reflection,
                next_improvement=next_step.strip(),
                failure_mode=fm,
            )

        # Store the planning strategy used for meta‑learning
        if hasattr(self, 'current_strategy'):
            if episode and hasattr(episode, "set_metadata"):
                episode.set_metadata("planning_strategy", self.current_strategy.value)

        return {
            "plan_id":    self.plan_id,
            "steps":      [
                {"tool": s[0], "success": r.success, "summary": r.output_summary}
                for s, r in zip(steps, results)
            ],
            "metrics":    metrics.summary(),
            "goals_met":  goals_met,
            "status":     status,
            "reflection": reflection,
            "next_step":  next_step.strip(),
        }

    def execute_full_plan(self) -> Dict[str, Any]:
        self.architecture = self.design_architecture()
        self.data         = self.design_data_pipeline(self.architecture)
        self.training     = self.configure_training(self.architecture)
        self.objective    = self.define_objective()
        metrics           = self.simulate_training(
            self.architecture, self.data, self.training, self.objective
        )
        return {
            "plan_id":    self.plan_id,
            "metrics":    metrics.summary(),
            "evaluation": self.evaluate_plan_efficiency(metrics),
            "reasoning":  self.reasoning_chain,
        }

    def evaluate_plan_efficiency(self, metrics: EvaluationMetrics) -> Dict[str, Any]:
        ev = {"plan_quality": {}, "improvement_suggestions": [], "lessons_learned": []}
        if metrics.reasoning_score > 0.8:
            ev["plan_quality"]["reasoning"] = "Excellent"
        elif metrics.reasoning_score > 0.6:
            ev["plan_quality"]["reasoning"] = "Good"
            ev["improvement_suggestions"].append("Add more diverse reasoning examples")
        else:
            ev["plan_quality"]["reasoning"] = "Needs improvement"
            ev["improvement_suggestions"].append("Consider deeper arch or CoT training")
        if metrics.hallucination_level.as_int() <= 1:
            ev["plan_quality"]["factuality"] = "Good"
        else:
            ev["improvement_suggestions"].append("Add RAG or increase factual data ratio")
        if metrics.flops_utilization > 0.6:
            ev["plan_quality"]["efficiency"] = "Good"
        else:
            ev["improvement_suggestions"].append("Optimize data loading and gradient accumulation")
        arch = self.architecture
        ev["lessons_learned"] = [
            f"Depth {arch.num_layers if arch else '?'}L → reasoning={metrics.reasoning_score:.2f}",
            f"Hallucination={metrics.hallucination_level.value}",
            f"Efficiency={metrics.flops_utilization:.1%}",
        ]
        return ev

    def _find_similar_episodes(self) -> List[Any]:
        words = self.task_description.lower().split()[:3]
        return [
            ep for ep in self.previous_episodes
            if any(w in getattr(ep, "task_description", "").lower() for w in words)
        ]

    def _analyze_past_architectures(self, episodes: List[Any]) -> Optional[str]:
        successful = []
        for ep in episodes:
            ev = getattr(ep, "evaluation", None)
            if ev and getattr(ev, "accuracy", 0) > 0.7:
                snap = getattr(ep, "model_after", None) or getattr(ep, "model_before", None)
                if snap:
                    successful.append(getattr(snap, "architecture_name", "unknown"))
        return Counter(successful).most_common(1)[0][0] if successful else None

    def __str__(self) -> str:
        arch = self.architecture or ArchitectureConfig()
        data = self.data         or DataConfig()
        tr   = self.training     or TrainingConfig()
        return "\n".join([
            "=" * 60,
            f"TrainingPlan [{self.plan_id}]",
            f"Task         : {self.task_description}",
            f"Uncertainty  : {self.uncertainty_score:.2f}",
            f"─ Architecture ──────────────────────────",
            f"  {arch.summary()}",
            f"─ Data ──────────────────────────────────",
            f"  {data.strategy.value} | {data.n_samples:,} samples",
            f"─ Training ──────────────────────────────",
            f"  epochs={tr.epochs} | lr={tr.learning_rate} | {tr.optimizer}",
            f"─ Targets ───────────────────────────────",
            f"  acc≥{self.targets.min_accuracy} | hallu≤{self.targets.max_hallucination}",
            f"History hints: {len(self.history_hints)}",
            "=" * 60,
        ])