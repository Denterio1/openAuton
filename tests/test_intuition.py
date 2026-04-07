"""
tests/test_intuition.py
=======================
Unit tests for core/intuition.py (IntuitionEngine, FeasibilityPredictor, RiskAnticipator, MentalCostSimulator)
"""

import sys
import pytest
from pathlib import Path
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.intuition import (
    IntuitionEngine,
    FeasibilityPredictor,
    RiskAnticipator,
    MentalCostSimulator,
    IntuitionScore,
    PlanIntuition,
)
from src.experience.episodes import ExperienceEpisode, TaskType, EpisodeStatus, EvaluationResult
from src.genome.dna import CognitiveDNA, Gene, GeneType


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------

@pytest.fixture
def sample_history():
    """Create a list of sample episodes for testing."""
    episodes = []
    for i in range(3):
        ep = Mock(spec=ExperienceEpisode)
        ep.task_type = TaskType.TRAIN_FROM_SCRATCH
        ep.tool_calls = []
        # Mock some successful tool calls
        if i > 0:
            call = Mock()
            call.tool_name = "run_training_job"
            call.success = True
            call.duration_seconds = 10.0
            ep.tool_calls.append(call)
        ep.status = EpisodeStatus.SUCCESS if i % 2 == 0 else EpisodeStatus.PARTIAL
        ep.evaluation = EvaluationResult(accuracy=0.8, plan_efficiency=0.7)
        episodes.append(ep)
    return episodes


@pytest.fixture
def sample_dna():
    """Create a sample cognitive DNA."""
    dna = CognitiveDNA(dna_id="test_dna", genes=[])
    gene = Gene(
        gene_id="g1",
        name="use_history",
        gene_type=GeneType.META_STRATEGY,
        value={"history_weight": 0.7},
        confidence=0.8,
    )
    dna.add_gene(gene)
    return dna


@pytest.fixture
def intuition_engine(sample_history, sample_dna):
    return IntuitionEngine(sample_history, sample_dna)


# ------------------------------------------------------------------
# Tests for FeasibilityPredictor
# ------------------------------------------------------------------

def test_feasibility_predictor_score(sample_history):
    predictor = FeasibilityPredictor(sample_history)
    # Test step scoring
    score = predictor.score_step(
        step_index=0,
        tool_name="run_training_job",
        tool_params={"epochs": 10, "batch_size": 32},
        task_type=TaskType.TRAIN_FROM_SCRATCH
    )
    assert isinstance(score, IntuitionScore)
    assert 0.0 <= score.intuition_score <= 1.0
    assert score.step_index == 0
    assert score.step_tool == "run_training_job"


def test_feasibility_predictor_no_history():
    predictor = FeasibilityPredictor([])  # empty history
    score = predictor.score_step(0, "run_training_job", {"epochs": 10}, TaskType.GENERAL)
    assert 0.0 <= score.intu_score <= 1.0
    assert "no history" in score.reason.lower()


# ------------------------------------------------------------------
# Tests for RiskAnticipator
# ------------------------------------------------------------------

def test_risk_anticipator(sample_history):
    anticipator = RiskAnticipator(sample_history)
    risk, fallback = anticipator.predict_risk("run_training_job", {"epochs": 100})
    # High epochs may trigger high_cost risk
    assert risk in ["high_cost", None]
    if risk == "high_cost":
        assert "reduce epochs" in (fallback or "")


def test_risk_anticipator_unknown_tool(sample_history):
    anticipator = RiskAnticipator(sample_history)
    risk, fallback = anticipator.predict_risk("unknown_tool", {})
    assert risk is None
    assert fallback is None


# ------------------------------------------------------------------
# Tests for MentalCostSimulator
# ------------------------------------------------------------------

def test_cost_simulator(sample_history):
    simulator = MentalCostSimulator(sample_history)
    cost, time_sec = simulator.estimate_step("run_training_job", {"epochs": 10})
    assert cost >= 0
    assert time_sec > 0


def test_cost_simulator_defaults():
    simulator = MentalCostSimulator([])
    cost, time_sec = simulator.estimate_step("run_training_job", {"epochs": 10})
    assert cost > 0
    assert time_sec > 0


# ------------------------------------------------------------------
# Tests for IntuitionEngine (main)
# ------------------------------------------------------------------

def test_intuition_engine_plan_evaluation(intuition_engine):
    plan_steps = [
        {"tool": "run_training_job", "params": {"epochs": 10}},
        {"tool": "evaluate_model", "params": {}},
    ]
    result = intuition_engine.evaluate_plan(
        plan_steps=plan_steps,
        task_type=TaskType.TRAIN_FROM_SCRATCH,
        budget_usd=1.0
    )
    assert isinstance(result, PlanIntuition)
    assert 0.0 <= result.overall_score <= 1.0
    assert result.estimated_cost_usd >= 0
    assert isinstance(result.recommended, bool)
    assert len(result.step_scores) == len(plan_steps)


def test_intuition_engine_rejects_high_cost_plan(intuition_engine):
    # Plan with many expensive steps
    plan_steps = [
        {"tool": "run_training_job", "params": {"epochs": 1000}},
        {"tool": "run_training_job", "params": {"epochs": 1000}},
    ]
    result = intuition_engine.evaluate_plan(
        plan_steps=plan_steps,
        task_type=TaskType.TRAIN_FROM_SCRATCH,
        budget_usd=0.5
    )
    # Estimated cost may exceed budget, so recommended may be False
    if result.estimated_cost_usd > 0.5:
        assert result.recommended is False
        assert result.warning is not None


def test_intuition_engine_missing_tool(intuition_engine):
    plan_steps = [{"tool": "non_existent_tool", "params": {}}]
    result = intuition_engine.evaluate_plan(plan_steps, TaskType.GENERAL)
    # The intuition engine should still produce a score (low confidence)
    assert result.overall_score >= 0
    assert result.step_scores[0].intuition_score < 0.5  # unknown tool, low confidence


def test_intuition_engine_empty_plan(intuition_engine):
    result = intuition_engine.evaluate_plan([], TaskType.GENERAL)
    assert result.overall_score == 0.0
    assert result.estimated_cost_usd == 0.0
    assert result.recommended is False  # empty plan not recommended


# ------------------------------------------------------------------
# Run with: pytest tests/test_intuition.py -v
# ------------------------------------------------------------------

if __name__ == "__main__":
    pytest.main([__file__, "-v"])