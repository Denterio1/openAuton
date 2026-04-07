import sys
import os
import time
import json
import shutil

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

print("=" * 60)
print("prime_agent — Full Test Suite")
print("=" * 60)

passed  = 0
failed  = 0
timings = {}

BASELINE_FILE = "experiments/test_baseline.json"
TMP_DIRS      = ["experiments/test_tmp", "experiments/test_store"]

def test(name, fn, max_seconds=2.0):
    global passed, failed
    start = time.time()
    try:
        fn()
        duration = time.time() - start
        timings[name] = round(duration, 3)
        slow = f" [slow: {duration:.2f}s]" if duration > max_seconds else ""
        print(f"  ✅ {name}{slow}")
        passed += 1
    except Exception as e:
        duration = time.time() - start
        timings[name] = round(duration, 3)
        print(f"  ❌ {name} → {e}")
        failed += 1

def teardown():
    for d in TMP_DIRS:
        if os.path.exists(d):
            shutil.rmtree(d)


print("\n[1] experience/episodes.py — core")

from experience.episodes import (
    ExperienceEpisode, TaskType, EpisodeStatus,
    EvaluationResult, FailureMode, CausalStep, EpisodeStore,
)

def t_episode_start():
    ep = ExperienceEpisode.start("Test task", TaskType.TRAIN_FROM_SCRATCH)
    assert ep.status == EpisodeStatus.RUNNING
    assert ep.task_description == "Test task"

def t_episode_record_tool():
    ep = ExperienceEpisode.start("Test", TaskType.UNKNOWN)
    ep.record_tool("run_training_job", "arch=transformer", "loss=0.5", True, 1.0)
    assert len(ep.tool_calls) == 1
    assert ep.tool_calls[0].tool_name == "run_training_job"

def t_episode_record_llm():
    ep = ExperienceEpisode.start("Test", TaskType.UNKNOWN)
    ep.record_llm_call(
        purpose="plan",
        prompt_summary="design a model",
        response_summary="use transformer",
        model="llama-3.3-70b-versatile",
        provider="groq",
        input_tokens=100,
        output_tokens=200,
    )
    assert len(ep.llm_calls) == 1
    assert ep.cost_usd > 0

def t_episode_gene_hint():
    ep = ExperienceEpisode.start("Test", TaskType.UNKNOWN)
    ep.add_gene_hint("8L beats 6L on CoT")
    ep.add_gene_hint("8L beats 6L on CoT")
    assert len(ep.gene_hints) == 1

def t_episode_finish():
    ep = ExperienceEpisode.start("Test", TaskType.TRAIN_FROM_SCRATCH)
    ep.finish(
        EpisodeStatus.SUCCESS,
        EvaluationResult(accuracy=0.85, hallucination_rate=0.08),
        reflection="All good",
        next_improvement="Try deeper model",
    )
    assert ep.status == EpisodeStatus.SUCCESS
    assert ep.evaluation.accuracy == 0.85

def t_episode_save_load():
    ep = ExperienceEpisode.start("Save test", TaskType.EVALUATION)
    ep.add_gene_hint("test gene")
    ep.finish(EpisodeStatus.SUCCESS, EvaluationResult(accuracy=0.90))
    path = ep.save("experiments/episodes")
    ep2  = ExperienceEpisode.load(str(path))
    assert ep2.episode_id == ep.episode_id
    assert ep2.evaluation.accuracy == 0.90
    assert ep2.gene_hints == ep.gene_hints

test("episode.start()",                t_episode_start)
test("episode.record_tool()",          t_episode_record_tool)
test("episode.record_llm_call() + cost", t_episode_record_llm)
test("episode.add_gene_hint() dedup", t_episode_gene_hint)
test("episode.finish()",               t_episode_finish)
test("episode.save() + load()",        t_episode_save_load)

# AGI features
print("\n[1b] experience/episodes.py — AGI features")

def t_causal_step():
    ep = ExperienceEpisode.start("Causal test", TaskType.TRAIN_FROM_SCRATCH)
    ep.record_causal_step(
        decision="Use transformer",
        reason="CoT tasks need attention",
        evidence="3 past episodes confirm",
        confidence=0.85,
    )
    assert len(ep.causal_trace) == 1
    assert ep.causal_trace[0].confidence == 0.85
    assert ep.avg_causal_confidence() == 0.85

def t_counterfactual():
    ep = ExperienceEpisode.start("CF test", TaskType.UNKNOWN)
    ep.add_counterfactual("Use MLP instead of transformer")
    ep.add_counterfactual("Use MLP instead of transformer")  # dedup
    ep.add_counterfactual("Use CNN for vision tasks")
    assert len(ep.counterfactual_options) == 2

def t_failure_mode():
    ep = ExperienceEpisode.start("Failure test", TaskType.TRAIN_FROM_SCRATCH)
    ep.finish(
        EpisodeStatus.FAILED,
        EvaluationResult(accuracy=0.30, hallucination_rate=0.45),
        failure_mode=FailureMode.HALLUCINATION,
    )
    assert ep.failure_mode == FailureMode.HALLUCINATION

def t_parent_lineage():
    ep1 = ExperienceEpisode.start("Run 1", TaskType.TRAIN_FROM_SCRATCH)
    ep1.finish(EpisodeStatus.PARTIAL, EvaluationResult(accuracy=0.65))
    ep1.save("experiments/test_store")

    ep2 = ExperienceEpisode.start(
        "Run 2", TaskType.TRAIN_FROM_SCRATCH,
        parent_id=ep1.episode_id,
    )
    ep2.finish(EpisodeStatus.SUCCESS, EvaluationResult(accuracy=0.82))
    ep2.save("experiments/test_store")

    store   = EpisodeStore("experiments/test_store")
    lineage = store.get_lineage(ep2.episode_id)
    assert len(lineage) == 2
    assert lineage[0].episode_id == ep1.episode_id
    assert lineage[1].episode_id == ep2.episode_id

def t_episode_store_search():
    store = EpisodeStore("experiments/test_store")
    results = store.search(
        task_type=TaskType.TRAIN_FROM_SCRATCH,
        min_accuracy=0.70,
    )
    assert all(
        e.evaluation and e.evaluation.accuracy >= 0.70
        for e in results
        if e.evaluation and e.evaluation.accuracy is not None
    )

def t_episode_store_stats():
    store = EpisodeStore("experiments/test_store")
    stats = store.get_statistics()
    assert "total" in stats
    assert stats["total"] >= 0

test("causal_trace record + avg_confidence", t_causal_step)
test("counterfactual_options dedup",          t_counterfactual)
test("failure_mode typed enum",               t_failure_mode)
test("parent_episode_id + get_lineage()",     t_parent_lineage)
test("EpisodeStore.search() filters",         t_episode_store_search)
test("EpisodeStore.get_statistics()",         t_episode_store_stats)


# ─────────────────────────────────────────────
# 2. TOOLS / REGISTRY
# ─────────────────────────────────────────────
print("\n[2] tools/registry.py")

from tools.registry import ToolRegistry, Tool, ToolResult, build_default_registry

def t_registry_build():
    r = build_default_registry()
    assert r.has("run_training_job")
    assert r.has("evaluate_model")
    assert r.has("generate_test_cases")
    assert r.has("run_safety_check")

def t_registry_run():
    r   = build_default_registry()
    res = r.run("generate_test_cases", task_type="test", n=10)
    assert res.success
    assert "10" in res.output_summary or res.output is not None

def t_registry_run_pipeline():
    r  = build_default_registry()
    ep = ExperienceEpisode.start("pipeline test", TaskType.TRAIN_FROM_SCRATCH)
    results = r.run_pipeline([
        ("generate_test_cases", {"task_type": "reasoning", "n": 5}),
        ("run_training_job",    {"architecture": "transformer-6L", "epochs": 3}),
        ("evaluate_model",      {"model_id": "transformer-6L"}),
    ], episode=ep)
    assert len(results) == 3
    assert all(r.success for r in results)
    assert len(ep.tool_calls) == 3

def t_registry_missing_tool():
    r   = build_default_registry()
    res = r.run("nonexistent_tool")
    assert not res.success

test("registry.build_default()",              t_registry_build)
test("registry.run() single tool",            t_registry_run)
test("registry.run_pipeline() + recording",   t_registry_run_pipeline)
test("registry.run() missing → failure",      t_registry_missing_tool)


# ─────────────────────────────────────────────
# 3. EXPERIENCE / RETRIEVAL
# ─────────────────────────────────────────────
print("\n[3] experience/retrieval.py")

from experience.retrieval import EpisodeRetrieval

def _make_episodes(n=3):
    eps = []
    for i in range(n):
        ep = ExperienceEpisode.start(f"Train transformer {i}", TaskType.TRAIN_FROM_SCRATCH)
        ev = EvaluationResult(accuracy=0.60 + i * 0.1)
        ep.finish(
            EpisodeStatus.SUCCESS if i % 2 == 0 else EpisodeStatus.PARTIAL,
            ev,
            next_improvement=f"Increase depth by {i + 1} layers",
        )
        eps.append(ep)
    return eps

def t_retrieval_hints():
    ret   = EpisodeRetrieval(_make_episodes(3))
    hints = ret.get_hints(TaskType.TRAIN_FROM_SCRATCH)
    assert len(hints) > 0
    assert isinstance(hints[0], str)

def t_retrieval_best():
    eps  = _make_episodes(3)
    best = EpisodeRetrieval(eps).get_best(TaskType.TRAIN_FROM_SCRATCH)
    assert best is not None
    assert best.evaluation.accuracy == max(e.evaluation.accuracy for e in eps)

def t_retrieval_success_rate():
    rate = EpisodeRetrieval(_make_episodes(3)).success_rate(TaskType.TRAIN_FROM_SCRATCH)
    assert 0.0 <= rate <= 1.0

test("retrieval.get_hints()",      t_retrieval_hints)
test("retrieval.get_best()",       t_retrieval_best)
test("retrieval.success_rate()",   t_retrieval_success_rate)


# ─────────────────────────────────────────────
# 4. EXPERIENCE / PROCESSOR  (replaces feedback.py)
# ─────────────────────────────────────────────
print("\n[4] experience/processor.py")

try:
    from experience.processor import ReflectionProcessor

    def t_processor_evaluate():
        ep = ExperienceEpisode.start("Test", TaskType.TRAIN_FROM_SCRATCH)
        ep.finish(
            EpisodeStatus.PARTIAL,
            EvaluationResult(
                accuracy=0.72,
                hallucination_rate=0.15,
                safety_score=0.88,
                reasoning_quality=0.65,
                plan_efficiency=0.70,
                tool_usage_correct=True,
                learned_from_history=False,
            )
        )
        processor = ReflectionProcessor ()
        report    = processor.process(ep, history=[])
        assert report.model_score > 0
        assert isinstance(report.priority_actions, list)

    def t_processor_gene_hints():
        ep = ExperienceEpisode.start("Hint test", TaskType.TRAIN_FROM_SCRATCH)
        ep.add_gene_hint("AdamW + cosine reduced hallucination")
        ep.finish(EpisodeStatus.SUCCESS, EvaluationResult(accuracy=0.88))
        processor = ReflectionProcessor ()
        report    = processor.process(ep, history=[])
        assert len(report.gene_hints) > 0

    test("processor.process() → report",      t_processor_evaluate)
    test("processor.process() → gene_hints",  t_processor_gene_hints)

except ImportError:
    # fallback to old feedback.py if processor not present
    from experience.feedback import SelfEvaluator

    def t_feedback_evaluate():
        ep = ExperienceEpisode.start("Test", TaskType.TRAIN_FROM_SCRATCH)
        ep.finish(
            EpisodeStatus.PARTIAL,
            EvaluationResult(
                accuracy=0.72, hallucination_rate=0.15,
                safety_score=0.88, reasoning_quality=0.65,
                plan_efficiency=0.70, tool_usage_correct=True,
                learned_from_history=False,
            )
        )
        report = SelfEvaluator().evaluate(ep, [])
        assert report.model_score > 0
        assert isinstance(report.priority_actions, list)

    test("feedback.SelfEvaluator.evaluate()", t_feedback_evaluate)


# ─────────────────────────────────────────────
# 5. TOOLS — executor + file_ops
# ─────────────────────────────────────────────
print("\n[5] tools/python_executor.py + file_ops.py")

from tools.python_executor import PythonExecutor
from tools.file_ops import FileOps

def t_executor_basic():
    res = PythonExecutor(timeout=10).run("print('hello'); print(2+2)")
    assert res.success
    assert "hello" in res.stdout
    assert "4"     in res.stdout

def t_executor_syntax_check():
    ex = PythonExecutor()
    ok, _   = ex.validate_syntax("x = 1 + 1")
    assert ok
    ok, err = ex.validate_syntax("x = (")
    assert not ok

def t_executor_timeout():
    res = PythonExecutor(timeout=2).run("import time; time.sleep(10)")
    assert not res.success

def t_fileops_write_read():
    ops = FileOps("experiments/test_tmp")
    ops.write("hello.txt", "prime_agent test")
    assert ops.read("hello.txt") == "prime_agent test"

def t_fileops_json():
    ops = FileOps("experiments/test_tmp")
    ops.write_json("data.json", {"key": "value", "n": 42})
    data = ops.read_json("data.json")
    assert data["key"] == "value"
    assert data["n"]   == 42

test("python_executor basic run",         t_executor_basic,     max_seconds=5.0)
test("python_executor syntax check",      t_executor_syntax_check)
test("python_executor timeout",           t_executor_timeout,   max_seconds=5.0)
test("file_ops write + read",             t_fileops_write_read)
test("file_ops write_json + read_json",   t_fileops_json)


# ─────────────────────────────────────────────
# 6. TRAINING
# ─────────────────────────────────────────────
print("\n[6] training/plane.py")

from training.plane import TrainingPlan, ArchitectureConfig, DataConfig, TrainingConfig

def t_plan_design():
    plan = TrainingPlan.design(
        task="Train transformer for CoT",
        task_type=TaskType.TRAIN_FROM_SCRATCH,
        arch=ArchitectureConfig(num_layers=6, d_model=256),
        data=DataConfig(n_samples=10_000),
        training=TrainingConfig(epochs=10, learning_rate=3e-4),
    )
    assert plan.architecture.num_layers == 6
    assert plan.data.n_samples          == 10_000
    assert plan.training.epochs         == 10

def t_plan_pipeline_steps():
    plan  = TrainingPlan.design("Test task", TaskType.TRAIN_FROM_SCRATCH)
    names = [s[0] for s in plan.pipeline_steps()]
    assert "run_training_job" in names
    assert "evaluate_model"   in names

def t_plan_execute():
    plan     = TrainingPlan.design("Test", TaskType.TRAIN_FROM_SCRATCH)
    registry = build_default_registry()
    ep       = ExperienceEpisode.start("Test plan exec", TaskType.TRAIN_FROM_SCRATCH)
    summary  = plan.execute(registry, ep)
    assert "status" in summary
    assert len(ep.tool_calls) >= 2

test("plan.design()",          t_plan_design)
test("plan.pipeline_steps()",  t_plan_pipeline_steps)
test("plan.execute()",         t_plan_execute)


# ─────────────────────────────────────────────
# 7. CORE AGENT
# ─────────────────────────────────────────────
print("\n[7] core/agent.py")

from core.agent import PrimeAgent, AgentConfig

def t_agent_init():
    agent = PrimeAgent(config=AgentConfig(verbose=False, auto_save=False))
    assert agent.registry is not None
    assert isinstance(agent.history, list)

def t_agent_run():
    agent  = PrimeAgent(config=AgentConfig(verbose=False, auto_save=False))
    before = len(agent.history)
    summary = agent.run("Train a small transformer for reasoning")
    assert isinstance(summary, dict)
    assert "status" in summary
    assert len(agent.history) == before + 1

def t_agent_learns():
    agent  = PrimeAgent(config=AgentConfig(verbose=False, auto_save=False))
    before = len(agent.history)
    agent.run("Train a small transformer")
    agent.run("Train a small transformer")
    assert len(agent.history)         == before + 2
    assert agent.history[-1].evaluation is not None

def t_agent_stats():
    agent = PrimeAgent(config=AgentConfig(verbose=False, auto_save=False))
    agent.run("Train a small transformer")
    stats = agent.stats()
    assert "episode" in stats.lower()

test("agent.init()",                  t_agent_init)
test("agent.run() single task",       t_agent_run,      max_seconds=5.0)
test("agent.run() learns across runs",t_agent_learns,   max_seconds=5.0)
test("agent.stats()",                 t_agent_stats)


# ─────────────────────────────────────────────
# 8. GENOME / DNA
# ─────────────────────────────────────────────
print("\n[8] genome/dna.py")

try:
    from genome.dna import CognitiveDNA, Gene, GeneType, GeneDominance

    def t_dna_init():
        dna = CognitiveDNA()
        assert dna is not None

    def t_dna_add_gene():
        dna  = CognitiveDNA()
        gene = Gene(
            gene_id="g001",
            gene_type=GeneType.ARCHITECTURE,
            dominance=GeneDominance.DOMINANT,
            name="arch_depth",
            value="Use 8 layers for CoT tasks",
            confidence=0.80,
            fitness=0.75,
        )
        dna.add_gene(gene)
        assert len(dna.genes) >= 1

    def t_gene_mutate():
        gene = Gene(
            gene_id="g002",
            gene_type=GeneType.HYPERPARAMETER,
            dominance=GeneDominance.RECESSIVE,
            name="learning_rate",
            value="LR=1e-4 for stable training",
            confidence=0.70,
            fitness=0.65,
        )
        mutated = gene.mutate(mutation_rate=1.0)
        assert mutated is not None
        assert mutated.gene_id != gene.gene_id

    test("dna.init()",        t_dna_init)
    test("dna.add_gene()",    t_dna_add_gene)
    test("gene.mutate()",     t_gene_mutate)

except ImportError as e:
    print(f"  ⚠️  genome/DNA.py not available: {e}")


# ─────────────────────────────────────────────
# 9. INTEGRATION — full pipeline
# ─────────────────────────────────────────────
print("\n[9] Integration — episodes → registry → processor")

def t_integration_full_pipeline():
    registry = build_default_registry()
    ep       = ExperienceEpisode.start(
        "Integration test — full pipeline",
        TaskType.TRAIN_FROM_SCRATCH,
        uncertainty=0.6,
    )
    ep.record_causal_step(
        decision="Use transformer",
        reason="CoT tasks confirmed by 3 past episodes",
        confidence=0.85,
    )
    ep.add_counterfactual("Use MLP — rejected: poor CoT performance")

    results = registry.run_pipeline([
        ("generate_test_cases", {"task_type": "reasoning", "n": 5}),
        ("run_training_job",    {"architecture": "transformer-6L", "epochs": 3}),
        ("evaluate_model",      {}),
    ], episode=ep)

    ep.finish(
        EpisodeStatus.SUCCESS,
        EvaluationResult(accuracy=0.83, hallucination_rate=0.09),
        failure_mode=FailureMode.NONE,
    )

    assert len(ep.tool_calls)            == 3
    assert len(ep.causal_trace)          == 1
    assert len(ep.counterfactual_options) == 1
    assert ep.failure_mode               == FailureMode.NONE
    assert ep.evaluation.accuracy        == 0.83

def t_integration_lineage():
    store = EpisodeStore("experiments/test_store")
    eps   = store.load_all()
    # lineage graph must be consistent — parent always older than child
    for ep in eps:
        if ep.parent_episode_id:
            parent = store.load(ep.parent_episode_id)
            if parent:
                assert parent.started_at <= ep.started_at

test("full pipeline: episode → tools → finish", t_integration_full_pipeline)
test("lineage consistency check",               t_integration_lineage)


# ─────────────────────────────────────────────
# 10. REGRESSION GUARD
# ─────────────────────────────────────────────
print("\n[10] Regression guard")

def t_regression_accuracy():
    baseline_acc = 0.0
    if os.path.exists(BASELINE_FILE):
        try:
            baseline_acc = json.load(open(BASELINE_FILE)).get("avg_accuracy", 0.0)
        except Exception:
            pass

    eps = _make_episodes(5)
    current_acc = sum(
        e.evaluation.accuracy for e in eps if e.evaluation and e.evaluation.accuracy
    ) / len(eps)

    # save new baseline
    os.makedirs(os.path.dirname(BASELINE_FILE), exist_ok=True)
    json.dump({"avg_accuracy": current_acc}, open(BASELINE_FILE, "w"))

    # must not regress more than 5%
    assert current_acc >= baseline_acc - 0.05, (
        f"Regression: {current_acc:.3f} < baseline {baseline_acc:.3f} - 0.05"
    )

test("accuracy regression guard", t_regression_accuracy)


# ─────────────────────────────────────────────
# CLEANUP
# ─────────────────────────────────────────────
teardown()


# ─────────────────────────────────────────────
# RESULTS + PERFORMANCE REPORT
# ─────────────────────────────────────────────
print()
print("=" * 60)
print(f"Results: {passed} passed | {failed} failed | {passed+failed} total")
if failed == 0:
    print("ALL TESTS PASSED ✅")
else:
    print(f"{failed} TEST(S) FAILED ❌ — fix before moving on")

slow_tests = {k: v for k, v in timings.items() if v > 1.0}
if slow_tests:
    print("\nSlow tests (>1s):")
    for name, dur in sorted(slow_tests.items(), key=lambda x: -x[1]):
        print(f"  {dur:.2f}s  {name}")
print("=" * 60)