"""
tools/human_input.py
====================
Human‑in‑the‑Loop Bridge – pauses agent execution, asks the user,
and resumes with user feedback integrated into the episode.

Components:
1. QueryDispatcher – formats clear, context‑aware questions for the user.
2. InterruptHandler – saves agent state, waits for input, then restores.
3. CriticalityFilter – decides whether human input is needed based on cost/risk.

Integration:
- The agent calls ask_human() when intuition_score is low or budget is high.
- The user's response is recorded in the episode and used for decision making.
- Supports console input (can be extended to web UI, Slack, etc.).
"""

from __future__ import annotations
import json
import pickle
import time
from pathlib import Path
from typing import Any, Dict, Optional, Callable, Union
from dataclasses import dataclass, field
from datetime import datetime

# ──────────────────────────────────────────────────────────
# Data structures for user interaction
# ──────────────────────────────────────────────────────────

@dataclass
class HumanQuery:
    """A question posed to the human user."""
    id: str
    context: str                # what the agent is doing
    problem: str                # what issue occurred
    options: list[str]          # possible choices (or "free text")
    urgency: str                # "low", "medium", "high"
    cost_estimate: float        # USD if relevant
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def format_for_display(self) -> str:
        lines = [
            "=" * 60,
            f"[HUMAN INPUT REQUIRED] – {self.urgency.upper()} urgency",
            f"Context: {self.context}",
            f"Problem: {self.problem}",
            f"Estimated cost: ${self.cost_estimate:.4f}",
            "Options:",
        ]
        for i, opt in enumerate(self.options, 1):
            lines.append(f"  {i}. {opt}")
        lines.append("Your choice (number or free text): ")
        return "\n".join(lines)


# ──────────────────────────────────────────────────────────
# 1. QueryDispatcher – formulates clear questions
# ──────────────────────────────────────────────────────────

class QueryDispatcher:
    """
    Translates agent's internal state into a human‑readable question.
    """

    @staticmethod
    def dispatch(agent_state: Dict[str, Any], intuition_report: Optional[Dict] = None) -> HumanQuery:
        """
        Create a HumanQuery based on the agent's current situation.
        """
        # Default template
        context = agent_state.get("current_task", "Unknown task")
        problem = "The agent is uncertain about the next step."
        options = ["Proceed anyway", "Abort this task", "Adjust parameters and retry", "Provide free‑text instruction"]
        urgency = "medium"
        cost_est = agent_state.get("estimated_cost", 0.1)

        # Use intuition report if available
        if intuition_report:
            score = intuition_report.get("overall_score", 0.5)
            risk = intuition_report.get("warning", "")
            if score < 0.3:
                urgency = "high"
                problem = f"Intuition score is very low ({score:.2f}). {risk}"
                options.insert(0, "Fallback to safer plan")
            elif score < 0.6:
                urgency = "medium"
                problem = f"Intuition score is moderate ({score:.2f}). {risk}"
            else:
                urgency = "low"
                problem = f"Intuition score is acceptable ({score:.2f}). Do you want to proceed?"

        # Check cost threshold
        if cost_est > agent_state.get("budget_left", 1.0):
            urgency = "high"
            problem += f" Estimated cost ${cost_est:.2f} exceeds remaining budget."

        return HumanQuery(
            id=f"hq_{int(time.time())}",
            context=context,
            problem=problem,
            options=options,
            urgency=urgency,
            cost_estimate=cost_est,
        )


# ──────────────────────────────────────────────────────────
# 2. InterruptHandler – saves and restores agent state
# ──────────────────────────────────────────────────────────

class InterruptHandler:
    """
    Pauses agent execution, saves state to disk, waits for human input,
    then restores state and returns the user's response.
    """

    def __init__(self, state_dir: Path = Path("experiments/human_interrupts")):
        self.state_dir = state_dir
        self.state_dir.mkdir(parents=True, exist_ok=True)

    def save_state(self, state: Dict[str, Any], query_id: str) -> Path:
        """Save agent state to disk for later restoration."""
        state_path = self.state_dir / f"{query_id}_state.pkl"
        with open(state_path, "wb") as f:
            pickle.dump(state, f)
        return state_path

    def load_state(self, query_id: str) -> Dict[str, Any]:
        """Restore agent state from disk."""
        state_path = self.state_dir / f"{query_id}_state.pkl"
        with open(state_path, "rb") as f:
            return pickle.load(f)

    def cleanup(self, query_id: str):
        """Delete saved state after restoration."""
        state_path = self.state_dir / f"{query_id}_state.pkl"
        if state_path.exists():
            state_path.unlink()

    def wait_for_input(self, query: HumanQuery, timeout_seconds: Optional[float] = None) -> str:
        """
        Display query to user and wait for response.
        Returns the user's input (as string).
        """
        print(query.format_for_display())
        if timeout_seconds:
            # Simple timeout using input() – for production, use asyncio or threading
            import threading
            user_input = [None]
            def get_input():
                try:
                    user_input[0] = input("> ")
                except:
                    user_input[0] = ""
            t = threading.Thread(target=get_input)
            t.daemon = True
            t.start()
            t.join(timeout_seconds)
            if t.is_alive():
                print("\n[Timeout] No response, defaulting to option 1.")
                return "1"
            return user_input[0] if user_input[0] is not None else ""
        else:
            return input("> ")

    def process_response(self, response: str, query: HumanQuery) -> Dict[str, Any]:
        """
        Parse user response into a structured action.
        """
        # Try to interpret as number (option index)
        if response.isdigit():
            idx = int(response) - 1
            if 0 <= idx < len(query.options):
                chosen = query.options[idx]
                return {
                    "action": "select_option",
                    "choice": chosen,
                    "raw_response": response,
                }
        # Otherwise treat as free text instruction
        return {
            "action": "free_text",
            "instruction": response,
            "raw_response": response,
        }


# ──────────────────────────────────────────────────────────
# 3. CriticalityFilter – decides when to bother the human
# ──────────────────────────────────────────────────────────

class CriticalityFilter:
    """
    Determines whether human input is necessary based on:
    - intuition score (lower = more need)
    - cost estimate relative to budget
    - confidence in the plan
    - whether this is a critical decision point
    """

    def __init__(self, min_intuition_score: float = 0.4,
                 max_cost_ratio: float = 0.8,
                 ask_for_high_cost: bool = True):
        self.min_intuition_score = min_intuition_score
        self.max_cost_ratio = max_cost_ratio
        self.ask_for_high_cost = ask_for_high_cost

    def needs_human(self, agent_state: Dict[str, Any],
                    intuition_report: Optional[Dict] = None) -> Tuple[bool, str]:
        """
        Returns (needs_human, reason).
        """
        # 1. Check intuition score
        if intuition_report:
            score = intuition_report.get("overall_score", 0.5)
            if score < self.min_intuition_score:
                return True, f"Intuition score {score:.2f} below threshold {self.min_intuition_score}"

        # 2. Check cost
        est_cost = agent_state.get("estimated_cost", 0.0)
        budget = agent_state.get("budget_left", 1.0)
        if self.ask_for_high_cost and est_cost > budget * self.max_cost_ratio:
            return True, f"Estimated cost ${est_cost:.2f} exceeds {self.max_cost_ratio:.0%} of remaining budget"

        # 3. Check if this is a first‑time decision (no history)
        if len(agent_state.get("history", [])) == 0:
            return True, "No prior episodes – human guidance needed for first run"

        # 4. Check if previous attempt failed for the same task
        recent_failures = [ep for ep in agent_state.get("history", [])[-3:] if ep.status != "success"]
        if len(recent_failures) >= 2:
            return True, f"Multiple recent failures ({len(recent_failures)}) – human advice required"

        return False, "All criteria satisfied – can proceed autonomously"


# ──────────────────────────────────────────────────────────
# 4. Main HumanInput Tool (for registry)
# ──────────────────────────────────────────────────────────

class HumanInputTool:
    """
    The main tool that the agent calls when it needs human input.
    It suspends execution, asks the user, and returns the decision.
    """

    def __init__(self, state_dir: Path = Path("experiments/human_interrupts")):
        self.dispatcher = QueryDispatcher()
        self.interrupt = InterruptHandler(state_dir)
        self.filter = CriticalityFilter()
        self._last_response: Optional[Dict] = None

    def ask(self, agent_state: Dict[str, Any],
            intuition_report: Optional[Dict] = None,
            force: bool = False) -> Dict[str, Any]:
        """
        Main entry point.
        If force=True, always ask; otherwise check criticality filter.
        Returns a dict with 'action' and user's choice/instruction.
        """
        if not force:
            needed, reason = self.filter.needs_human(agent_state, intuition_report)
            if not needed:
                # No need to interrupt – return a default "proceed" response
                return {
                    "action": "auto_proceed",
                    "reason": reason,
                    "response": "proceed automatically"
                }

        # Create query
        query = self.dispatcher.dispatch(agent_state, intuition_report)

        # Save current agent state (simulated – in real agent, you'd pass the full state)
        # For now, we just store a minimal representation
        state_snapshot = {
            "agent_state": agent_state,
            "intuition": intuition_report,
            "timestamp": datetime.now().isoformat(),
        }
        self.interrupt.save_state(state_snapshot, query.id)

        # Wait for user input
        raw_response = self.interrupt.wait_for_input(query, timeout_seconds=None)

        # Parse response
        result = self.interrupt.process_response(raw_response, query)

        # Cleanup saved state
        self.interrupt.cleanup(query.id)

        self._last_response = result
        return result

    def get_last_response(self) -> Optional[Dict]:
        """Return the most recent user response."""
        return self._last_response


# ──────────────────────────────────────────────────────────
# Factory function for registry integration
# ──────────────────────────────────────────────────────────

def create_human_input_tool():
    """
    Returns a callable that can be registered as a tool.
    """
    tool = HumanInputTool()

    def human_input_function(agent_state: Dict[str, Any],
                             intuition_score: Optional[float] = None,
                             force: bool = False) -> Dict[str, Any]:
        """
        Tool signature for registry.
        """
        intuition_report = None
        if intuition_score is not None:
            intuition_report = {"overall_score": intuition_score}
        return tool.ask(agent_state, intuition_report, force)

    return human_input_function
