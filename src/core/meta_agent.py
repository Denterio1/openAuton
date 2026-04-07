"""
core/meta_agent.py
==================
Hierarchical Meta-Cognition for openAuton 2.0.

Components:
- TaskAgent: Executes specific tasks (code writing, data analysis, etc.)
- MetaAgent: Observes TaskAgent and suggests improvements at the meta-level.
- MetaLearningLoop: Orchestrates the interaction between TaskAgent and MetaAgent.
"""

from __future__ import annotations
import json
import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field

from ..llm.provider import ModelProvider
from ..experience.episodes import ExperienceEpisode
from ..genome.dna import CognitiveDNA
from .intuition import IntuitionEngine
from ..tools.registry import ToolRegistry

logger = logging.getLogger(__name__)


@dataclass
class MetaSuggestion:
    """Suggestion from the MetaAgent to improve the TaskAgent."""
    component: str  # "planner", "tools", "strategy", "dna"
    change: Dict[str, Any]
    reasoning: str
    confidence: float


class TaskAgent:
    """
    Specialized agent for executing specific tasks.
    This is essentially the PrimeAgent but with a focus on execution.
    """

    def __init__(self, llm: ModelProvider, registry: ToolRegistry, dna: CognitiveDNA):
        self.llm = llm
        self.registry = registry
        self.dna = dna
        self.intuition = IntuitionEngine([], dna)  # history will be loaded later

    def execute(self, task: str, context: str = "") -> Dict[str, Any]:
        """
        Execute a task using the current plan (dynamic or fallback).
        """
        # 1. Generate a dynamic plan using LLM
        plan = self._plan_with_llm(task, context)
        if not plan:
            # Fallback to default tools
            plan = self._default_plan(task)

        # 2. Evaluate plan with intuition engine
        accepted, intuition = self.intuition.evaluate_plan(plan, "general")
        if not accepted:
            logger.warning(f"Plan rejected by intuition: {intuition.warning}")
            # Could fall back to a simpler plan or ask human
            return {"status": "rejected", "reason": intuition.warning}

        # 3. Execute plan
        results = []
        for step in plan:
            tool_name = step.get("tool")
            params = step.get("params", {})
            result = self.registry.run(tool_name, **params)
            results.append({
                "tool": tool_name,
                "success": result.success,
                "output": result.output_summary,
            })
            if not result.success:
                break

        return {
            "status": "success" if all(r["success"] for r in results) else "partial",
            "steps": results,
            "plan": plan,
        }

    def _plan_with_llm(self, task: str, context: str) -> List[Dict]:
        """
        Generate a dynamic plan using the LLM.
        """
        tools_desc = self.registry.describe_all()
        prompt = f"""You are a planning engine. Given the task and available tools, create a step‑by‑step plan.

Task: {task}
Context: {context}

Available tools:
{tools_desc}

Return ONLY valid JSON with the following structure:
{{
    "steps": [
        {{"tool": "tool_name", "reason": "why this step", "params": {{}}}},
        ...
    ]
}}"""
        try:
            response = self.llm.think(prompt, temperature=0.2)
            data = json.loads(response)
            return data.get("steps", [])
        except Exception as e:
            logger.error(f"LLM planning failed: {e}")
            return []

    def _default_plan(self, task: str) -> List[Dict]:
        """Fallback plan using default tools."""
        return [
            {"tool": "generate_test_cases", "params": {"task_type": "general", "n": 50}},
            {"tool": "run_training_job", "params": {"epochs": 10}},
            {"tool": "evaluate_model", "params": {}},
            {"tool": "run_safety_check", "params": {}},
        ]


class MetaAgent:
    """
    The meta-agent that observes the TaskAgent and suggests improvements.
    """

    def __init__(self, llm: ModelProvider):
        self.llm = llm
        self.suggestion_history: List[MetaSuggestion] = []

    def analyze_episode(self, episode: ExperienceEpisode) -> List[MetaSuggestion]:
        """
        Analyze an episode and generate meta-suggestions for improvement.
        """
        prompt = f"""You are a meta-learning agent. Analyze the following episode and suggest concrete improvements to the agent's architecture.

Episode ID: {episode.episode_id}
Task: {episode.task_description}
Status: {episode.status.value}
Metrics: accuracy={episode.evaluation.accuracy if episode.evaluation else 'N/A'}

Lessons learned: {episode.next_improvement if episode.next_improvement else 'None'}

For each suggestion, specify:
- component: one of ["planner", "tools", "strategy", "dna"]
- change: a JSON object describing the change
- reasoning: why this change would help
- confidence: float 0-1

Return a JSON array of suggestions.
"""
        try:
            response = self.llm.think(prompt, temperature=0.3)
            data = json.loads(response)
            suggestions = []
            for item in data:
                suggestions.append(MetaSuggestion(
                    component=item.get("component"),
                    change=item.get("change", {}),
                    reasoning=item.get("reasoning", ""),
                    confidence=item.get("confidence", 0.5),
                ))
            self.suggestion_history.extend(suggestions)
            return suggestions
        except Exception as e:
            logger.error(f"Meta analysis failed: {e}")
            return []

    def apply_suggestion(self, suggestion: MetaSuggestion, target: TaskAgent) -> bool:
        """
        Apply a meta-suggestion to the TaskAgent.
        """
        if suggestion.component == "planner":
            # Modify the planning prompt or logic
            logger.info(f"Applying planner suggestion: {suggestion.change}")
            # In a full implementation, this would update the planner's internal state
            return True
        elif suggestion.component == "dna":
            # Suggest a new gene to add to the DNA
            logger.info(f"Applying DNA suggestion: {suggestion.change}")
            # This would call target.dna.add_gene(...)
            return True
        elif suggestion.component == "tools":
            # Suggest adding or modifying a tool
            logger.info(f"Applying tool suggestion: {suggestion.change}")
            return True
        return False


class MetaLearningLoop:
    """
    Orchestrates the interaction between TaskAgent and MetaAgent.
    """

    def __init__(self, task_agent: TaskAgent, meta_agent: MetaAgent):
        self.task_agent = task_agent
        self.meta_agent = meta_agent
        self.episode_history: List[ExperienceEpisode] = []

    def run(self, task: str, context: str = "") -> Dict[str, Any]:
        """
        Execute a task, then analyze the episode and apply meta-suggestions.
        """
        # 1. Execute the task
        result = self.task_agent.execute(task, context)
        if result["status"] != "success":
            return result

        # 2. Create an episode from the execution
        # (In a real implementation, this would be more detailed)
        episode = ExperienceEpisode.start(task, "general")
        episode.finish("success", evaluation=None)  # simplified

        # 3. Analyze the episode with the meta-agent
        suggestions = self.meta_agent.analyze_episode(episode)

        # 4. Apply the top suggestions
        for suggestion in suggestions[:3]:  # apply top 3
            self.meta_agent.apply_suggestion(suggestion, self.task_agent)

        return result