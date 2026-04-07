from __future__ import annotations

import time
import random
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from tools.web_search import create_web_search_tool
from tools.research_rag import create_research_rag_tool
from tools.file_ops import FileOps


@dataclass
class ToolResult:
    success: bool
    output: Any
    output_summary: str
    duration_seconds: float = 0.0
    error_message: str = ""

    def short(self) -> str:
        status = "✓" if self.success else "✗"
        return f"{status} [{self.duration_seconds:.2f}s] {self.output_summary}"


@dataclass
class Tool:
    name: str
    description: str
    fn: Callable[..., ToolResult]
    tags: List[str] = field(default_factory=list)
    requires: List[str] = field(default_factory=list)

    def run(self, **kwargs) -> ToolResult:
        start = time.time()
        try:
            result = self.fn(**kwargs)
            result.duration_seconds = time.time() - start
            return result
        except Exception as exc:
            return ToolResult(
                success=False,
                output=None,
                output_summary=f"Exception in {self.name}",
                duration_seconds=time.time() - start,
                error_message=f"{type(exc).__name__}: {exc}",
            )


class ToolRegistry:

    def __init__(self) -> None:
        self._tools: Dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def unregister(self, name: str) -> None:
        self._tools.pop(name, None)

    def get(self, name: str) -> Optional[Tool]:
        return self._tools.get(name)

    def has(self, name: str) -> bool:
        return name in self._tools

    def list_tools(self, tag: Optional[str] = None) -> List[Tool]:
        tools = list(self._tools.values())
        if tag:
            tools = [t for t in tools if tag in t.tags]
        return tools

    def describe_all(self, tag: Optional[str] = None) -> str:
        lines = []
        for tool in self.list_tools(tag=tag):
            req = f" (requires: {', '.join(tool.requires)})" if tool.requires else ""
            lines.append(f"- {tool.name}: {tool.description}{req}")
        return "\n".join(lines)

    def run(self, name: str, episode=None, **kwargs) -> ToolResult:
        tool = self._tools.get(name)
        if tool is None:
            result = ToolResult(
                success=False,
                output=None,
                output_summary=f"Tool '{name}' not found.",
                error_message="ToolNotFound",
            )
        else:
            result = tool.run(**kwargs)
        if episode is not None:
            episode.record_tool(
                tool_name=name,
                input_summary=str(kwargs)[:200],
                output_summary=result.output_summary,
                success=result.success,
                duration=result.duration_seconds,
                error=result.error_message,
            )
        return result

    def run_pipeline(self, steps, episode=None, stop_on_failure=True):
        results = []
        for name, kwargs in steps:
            result = self.run(name, episode=episode, **kwargs)
            results.append(result)
            if not result.success and stop_on_failure:
                break
        return results

    def __len__(self): return len(self._tools)
    def __repr__(self): return f"ToolRegistry({len(self)} tools: {list(self._tools.keys())})"


def build_default_registry() -> ToolRegistry:
    # ------------------------------------------------------------
    # Simulation tools (existing)
    # ------------------------------------------------------------
    def run_training_job(architecture="transformer", epochs=10, lr=3e-4, data_description="synthetic", **_):
        ls = round(random.uniform(2.0, 3.5), 3)
        le = round(ls * random.uniform(0.2, 0.5), 3)
        return ToolResult(True, {"loss_start": ls, "loss_end": le}, f"arch={architecture} | loss {ls}→{le}")

    def evaluate_model(model_id="latest", test_set="default", **_):
        acc, hallu, rq = round(random.uniform(0.55,0.90),3), round(random.uniform(0.05,0.30),3), round(random.uniform(0.50,0.85),3)
        return ToolResult(True, {"accuracy":acc,"hallucination_rate":hallu,"reasoning_quality":rq}, f"acc={acc} | hallu={hallu} | rq={rq}")

    def generate_test_cases(task_type="reasoning", n=50, difficulty="medium", **_):
        return ToolResult(True, [{"id":i} for i in range(n)], f"Generated {n} {difficulty} {task_type} test cases")

    def run_safety_check(model_id="latest", checks=None, **_):
        checks = checks or ["toxicity","bias","jailbreak"]
        res    = {c: round(random.uniform(0.80,1.0),3) for c in checks}
        passed = all(v>=0.85 for v in res.values())
        return ToolResult(passed, res, "Safety: "+" | ".join(f"{k}={v}" for k,v in res.items()))

    # ------------------------------------------------------------
    # Web search tool (internet)
    # ------------------------------------------------------------
    def web_search_wrapper(query: str, max_results: int = 5, **kwargs):
        try:
            search_fn = create_web_search_tool()
            result_dict = search_fn(query, max_results=max_results)
            if result_dict.get("status") == "success":
                output_summary = f"Found {result_dict['num_results']} results for '{query}'"
                return ToolResult(True, result_dict, output_summary)
            else:
                return ToolResult(False, None, f"Search failed: {result_dict.get('message', 'unknown')}")
        except Exception as e:
            return ToolResult(False, None, f"Exception: {e}")

    # ------------------------------------------------------------
    # Research RAG tool (arXiv papers)
    # ------------------------------------------------------------
    def research_rag_wrapper(query: str, action: str = "retrieve", top_k: int = 5, **kwargs):
        try:
            rag_tool = create_research_rag_tool(FileOps())
            result = rag_tool(query, action, top_k)
            if result.get("status") == "success":
                if action == "retrieve":
                    count = len(result.get("results", []))
                elif action == "insights":
                    count = len(result.get("insights", {}).get("architecture_hints", []))
                else:
                    count = 1
                output_summary = f"RAG {action}: {count} items"
                return ToolResult(True, result, output_summary)
            else:
                return ToolResult(False, None, f"RAG failed: {result.get('message', 'unknown')}")
        except Exception as e:
            return ToolResult(False, None, f"Exception: {e}")

    # ------------------------------------------------------------
    # Register all tools
    # ------------------------------------------------------------
    r = ToolRegistry()
    r.register(Tool("run_training_job",   "Train a model (simulation).",                    run_training_job,    tags=["training"]))
    r.register(Tool("evaluate_model",     "Evaluate model accuracy/hallucination.",         evaluate_model,      tags=["evaluation"], requires=["run_training_job"]))
    r.register(Tool("generate_test_cases","Generate synthetic test cases.",                 generate_test_cases, tags=["evaluation","data"]))
    r.register(Tool("run_safety_check",   "Run safety checks (toxicity, bias, jailbreak).", run_safety_check,    tags=["safety"]))
    r.register(Tool("web_search",         "Search the internet for information or code.",  web_search_wrapper,  tags=["research"]))
    r.register(Tool("research_rag",       "Search arXiv papers and extract insights.",      research_rag_wrapper,tags=["research","rag"]))
    return r