"""
genome/evolution.py
===================
Evolution Engine for Self‑Improving AI Agent

Components:
1. GeneExtractor – extracts programmable genes from UnifiedReport (gene_hints, lessons, metrics)
2. EvolutionaryOperators – selection, mutation, crossover for genes
3. MemoryConsolidation – updates dna.py / agent_dna.yaml with version control
4. ImpactPredictor – predicts cost/performance impact before applying mutations (integrates with intuition.py)
5. EvolutionEngine – main class that evolves the agent's DNA based on episodes

Now with LLM-powered gene suggestions.
"""

from __future__ import annotations
import yaml
import json
import copy
import random
import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime

# Import from our modules
from experience.processor import UnifiedReport
from genome.dna import CognitiveDNA, Gene, GeneType, GeneDominance


# ──────────────────────────────────────────────────────────
# 1. Gene Extractor – from UnifiedReport to structured genes
# (unchanged)
# ──────────────────────────────────────────────────────────

class GeneExtractor:
    """
    Converts textual/numerical hints from UnifiedReport into structured Gene objects.
    Repeated successful hints increase confidence.
    """

    def __init__(self):
        self._gene_memory: Dict[str, List[float]] = {}  # gene_id -> list of confidence scores

    def extract(self, report: UnifiedReport) -> List[Gene]:
        """
        Extract genes from a single report.
        Uses gene_hints, lessons_learned, metrics, and counterfactuals.
        """
        genes = []
        timestamp = datetime.now()

        # 1. From explicit gene_hints (e.g., "high_accuracy_85", "low_hallucination")
        for hint in report.gene_hints:
            gene = self._parse_hint_to_gene(hint, report)
            if gene:
                genes.append(gene)

        # 2. From lessons_learned – convert to "avoid_..." or "prefer_..." genes
        for lesson in report.lessons_learned:
            gene = self._lesson_to_gene(lesson, report)
            if gene:
                genes.append(gene)

        # 3. From counterfactual options – what was rejected but might be useful
        for cf in report.counterfactual_options:
            gene = self._counterfactual_to_gene(cf, report)
            if gene:
                genes.append(gene)

        # 4. From metrics directly – threshold-based genes
        if report.accuracy is not None and report.accuracy > 0.85:
            genes.append(Gene(
                gene_id="",
                name="high_accuracy",
                gene_type=GeneType.META_STRATEGY,
                value={"target_accuracy": report.accuracy},
                confidence=min(1.0, report.accuracy),
                source_episodes=[report.episode_id],
            ))
        if report.hallucination_rate is not None and report.hallucination_rate < 0.05:
            genes.append(Gene(
                gene_id="",
                name="ultra_low_hallucination",
                gene_type=GeneType.TRAINING_SCHEDULE,
                value={"hallucination_budget": report.hallucination_rate},
                confidence=1.0 - report.hallucination_rate,
                source_episodes=[report.episode_id],
            ))

        # Update confidence based on history (if same gene seen before)
        for gene in genes:
            if gene.gene_id in self._gene_memory:
                # Average confidence across occurrences
                self._gene_memory[gene.gene_id].append(gene.confidence)
                gene.confidence = sum(self._gene_memory[gene.gene_id]) / len(self._gene_memory[gene.gene_id])
            else:
                self._gene_memory[gene.gene_id] = [gene.confidence]

        return genes

    def _parse_hint_to_gene(self, hint: str, report: UnifiedReport) -> Optional[Gene]:
        """Convert string hint like 'high_accuracy_85' to a Gene."""
        if hint.startswith("high_accuracy_"):
            try:
                acc = int(hint.split("_")[-1]) / 100
                return Gene(
                    gene_id="",
                    name="target_accuracy",
                    gene_type=GeneType.HYPERPARAMETER,
                    value={"min_accuracy": acc},
                    confidence=acc,
                    source_episodes=[report.episode_id],
                )
            except:
                pass
        elif hint == "low_hallucination":
            return Gene(
                gene_id="",
                name="hallucination_penalty",
                gene_type=GeneType.OBJECTIVE,
                value={"hallucination_weight": 0.3},
                confidence=0.9,
                source_episodes=[report.episode_id],
            )
        elif hint == "high_safety":
            return Gene(
                gene_id="",
                name="safety_filter",
                gene_type=GeneType.EVALUATION_STRATEGY,
                value={"safety_threshold": 0.95},
                confidence=0.85,
                source_episodes=[report.episode_id],
            )
        elif hint == "learned_from_history":
            return Gene(
                gene_id="",
                name="use_history",
                gene_type=GeneType.META_STRATEGY,
                value={"history_weight": 0.7},
                confidence=0.8,
                source_episodes=[report.episode_id],
            )
        return None

    def _lesson_to_gene(self, lesson: str, report: UnifiedReport) -> Optional[Gene]:
        """Convert lesson like 'Accuracy too low → increase model capacity' to a gene."""
        if "increase model capacity" in lesson.lower():
            return Gene(
                gene_id="",
                name="prefer_deeper_network",
                gene_type=GeneType.ARCHITECTURE,
                value={"depth_increment": 2},
                confidence=0.7,
                source_episodes=[report.episode_id],
            )
        if "RLHF" in lesson or "contrastive loss" in lesson:
            return Gene(
                gene_id="",
                name="use_contrastive_loss",
                gene_type=GeneType.OBJECTIVE,
                value={"loss_type": "contrastive"},
                confidence=0.75,
                source_episodes=[report.episode_id],
            )
        return None

    def _counterfactual_to_gene(self, cf: str, report: UnifiedReport) -> Optional[Gene]:
        """Convert counterfactual like 'Use deeper transformer' to a gene."""
        if "deeper transformer" in cf.lower():
            return Gene(
                gene_id="",
                name="depth_preference",
                gene_type=GeneType.ARCHITECTURE,
                value={"num_layers": 12},
                confidence=0.6,
                source_episodes=[report.episode_id],
                dominance=GeneDominance.RECESSIVE,
            )
        if "chain-of-thought" in cf.lower():
            return Gene(
                gene_id="",
                name="cot_supervision",
                gene_type=GeneType.TRAINING_SCHEDULE,
                value={"cot_weight": 0.2},
                confidence=0.65,
                source_episodes=[report.episode_id],
            )
        return None


# ──────────────────────────────────────────────────────────
# 2. Evolutionary Operators (unchanged)
# ──────────────────────────────────────────────────────────

class EvolutionaryOperators:
    @staticmethod
    def select(genes: List[Gene], top_k: int = 5, min_confidence: float = 0.6) -> List[Gene]:
        candidates = [g for g in genes if g.confidence >= min_confidence]
        candidates.sort(key=lambda g: g.confidence, reverse=True)
        return candidates[:top_k]

    @staticmethod
    def mutate(gene: Gene, mutation_rate: float = 0.1, stable_trend: bool = False) -> Gene:
        if stable_trend and random.random() < mutation_rate:
            new_gene = copy.deepcopy(gene)
            new_gene.value = {"experimental": random.choice(["deeper", "wider", "dropout_0.3"])}
            new_gene.confidence *= 0.8
            new_gene.metadata["mutated_from"] = gene.gene_id
            return new_gene
        return gene.mutate(mutation_rate)

    @staticmethod
    def crossover(gene1: Gene, gene2: Gene) -> Optional[Gene]:
        if gene1.gene_type != gene2.gene_type:
            return None
        return gene1.combine(gene2, method="blend")


# ──────────────────────────────────────────────────────────
# 3. Memory Consolidation (unchanged)
# ──────────────────────────────────────────────────────────

class MemoryConsolidation:
    def __init__(self, dna_path: Path = Path("config/agent_dna.yaml"),
                 snapshot_dir: Path = Path("experiments/dna_snapshots")):
        self.dna_path = dna_path
        self.snapshot_dir = snapshot_dir
        self.snapshot_dir.mkdir(parents=True, exist_ok=True)

    def load_current_dna(self) -> Dict[str, Any]:
        if not self.dna_path.exists():
            return {
                "genes": [],
                "version": "1.0",
                "last_updated": datetime.now().isoformat(),
            }
        with open(self.dna_path, 'r') as f:
            return yaml.safe_load(f)

    def save_dna(self, dna_data: Dict[str, Any]) -> None:
        dna_data["last_updated"] = datetime.now().isoformat()
        with open(self.dna_path, 'w') as f:
            yaml.dump(dna_data, f, default_flow_style=False)

    def take_snapshot(self, dna_data: Dict[str, Any], reason: str = "before_evolution") -> str:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        snapshot_id = f"{timestamp}_{hashlib.md5(str(dna_data).encode()).hexdigest()[:8]}"
        snapshot_path = self.snapshot_dir / f"{snapshot_id}_{reason}.yaml"
        with open(snapshot_path, 'w') as f:
            yaml.dump(dna_data, f)
        return snapshot_id

    def consolidate(self, new_genes: List[Gene], current_dna: Dict[str, Any]) -> Dict[str, Any]:
        existing = {g.get("name", ""): g for g in current_dna.get("genes", [])}
        for gene in new_genes:
            name = gene.name
            if name in existing:
                old_conf = existing[name].get("confidence", 0.5)
                new_conf = (old_conf + gene.confidence) / 2
                existing[name]["confidence"] = new_conf
                existing[name]["last_seen"] = datetime.now().isoformat()
                existing[name]["source_episodes"] = list(set(
                    existing[name].get("source_episodes", []) + gene.source_episodes
                ))
            else:
                existing[name] = gene.to_dict()
        current_dna["genes"] = list(existing.values())
        current_dna["version"] = str(float(current_dna.get("version", "1.0")) + 0.1)
        return current_dna


# ──────────────────────────────────────────────────────────
# 4. Impact Predictor (unchanged)
# ──────────────────────────────────────────────────────────

class ImpactPredictor:
    def __init__(self, intuition_module=None):
        self.intuition = intuition_module

    def predict(self, gene: Gene, history: List[UnifiedReport]) -> Dict[str, Any]:
        success_rate = gene.confidence
        predicted_cost_impact = "low" if success_rate > 0.7 else "medium"
        predicted_accuracy_impact = "positive" if success_rate > 0.6 else "unknown"
        if self.intuition:
            return self.intuition.predict_gene_impact(gene, history)
        return {
            "cost_impact": predicted_cost_impact,
            "accuracy_impact": predicted_accuracy_impact,
            "recommended": success_rate > 0.5,
        }


# ──────────────────────────────────────────────────────────
# 5. Evolution Engine – with LLM-powered suggestions
# ──────────────────────────────────────────────────────────

class EvolutionEngine:
    """
    Main engine that evolves the agent's DNA based on episode reports.
    Now optionally uses an LLM to suggest new genes.
    """

    def __init__(self,
                 dna_path: Path = Path("config/agent_dna.yaml"),
                 snapshot_dir: Path = Path("experiments/dna_snapshots"),
                 intuition_module=None,
                 llm=None):                     # NEW: optional LLM provider
        self.extractor = GeneExtractor()
        self.operators = EvolutionaryOperators()
        self.memory = MemoryConsolidation(dna_path, snapshot_dir)
        self.predictor = ImpactPredictor(intuition_module)
        self.llm = llm                         # store LLM for later use
        self.evolution_history: List[Dict] = []

    # ──────────────────────────────────────────────
    # NEW: LLM-powered gene suggestion
    # ──────────────────────────────────────────────

    async def _llm_suggest_genes(self, report: UnifiedReport) -> List[Gene]:
        if not self.llm:
            return []
        prompt = f"""You are an expert AI evolution engineer. Based on the following episode report, suggest 2-3 new "genes" (strategies) that could improve future performance.

    Episode ID: {report.episode_id}
    Task type: {report.task_type.value}
    Status: {report.status.value}
    Accuracy: {report.accuracy or 0:.2f} (target ≥0.8)
    Hallucination: {report.hallucination_rate or 0:.2f} (target ≤0.1)
    Reasoning quality: {report.reasoning_quality or 0:.2f}
    Safety score: {report.safety_score or 0:.2f}

    Lessons learned: {', '.join(report.lessons_learned[:2])}
    Gene hints so far: {', '.join(report.gene_hints[:3])}
    Counterfactuals considered: {', '.join(report.counterfactual_options[:2])}

    Return a JSON array of objects, each with:
    - name: short identifier
    - gene_type: one of ["architecture","hyperparameter","data_strategy","objective","training_schedule","evaluation_strategy","rag_strategy","meta_strategy"]
    - value: a dictionary
    - confidence: float 0-1
    - reasoning: why this gene would help

    Example:
    [
      {{
        "name": "use_swa",
        "gene_type": "training_schedule",
        "value": {{"swa_start": 0.7}},
        "confidence": 0.8,
        "reasoning": "Accuracy is plateauing, SWA can smooth convergence"
      }}
    ]
    Return ONLY valid JSON, no extra text."""
        try:
            # Use think() instead of chat()
            resp = self.llm.think(prompt)
            if not resp.success:
                return []
            content = resp.content.strip()
            # Remove markdown code fences
            if content.startswith("```json"):
                content = content[7:]
            if content.startswith("```"):
                content = content[3:]
            if content.endswith("```"):
                content = content[:-3]
            data = json.loads(content)
            if not isinstance(data, list):
                return []
            genes = []
            for item in data:
                gene_type_str = item.get("gene_type", "meta_strategy")
                gene_type_map = {
                    "architecture": GeneType.ARCHITECTURE,
                    "hyperparameter": GeneType.HYPERPARAMETER,
                    "data_strategy": GeneType.DATA_STRATEGY,
                    "objective": GeneType.OBJECTIVE,
                    "training_schedule": GeneType.TRAINING_SCHEDULE,
                    "evaluation_strategy": GeneType.EVALUATION_STRATEGY,
                    "rag_strategy": GeneType.RAG_STRATEGY,
                    "meta_strategy": GeneType.META_STRATEGY,
                }
                gene_type = gene_type_map.get(gene_type_str, GeneType.META_STRATEGY)
                gene = Gene(
                    gene_id="",
                    name=item.get("name", "llm_suggested"),
                    gene_type=gene_type,
                    value=item.get("value", {}),
                    confidence=min(1.0, max(0.0, item.get("confidence", 0.5))),
                    source_episodes=[report.episode_id],
                    metadata={"llm_reasoning": item.get("reasoning", "")}
                )
                genes.append(gene)
            return genes
        except Exception as e:
            print(f"LLM gene suggestion failed: {e}")
            return []
    # ──────────────────────────────────────────────
    # Modified evolve() – integrates LLM suggestions
    # ──────────────────────────────────────────────

    def evolve(self, report: UnifiedReport, current_dna: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Main evolution function.
        Takes a UnifiedReport, extracts genes, selects best, applies mutation/crossover,
        and optionally adds LLM‑suggested genes. Consolidates into DNA.
        """
        # 1. Load current DNA if not provided
        if current_dna is None:
            current_dna = self.memory.load_current_dna()

        # 2. Take snapshot before evolution
        snapshot_id = self.memory.take_snapshot(current_dna, "before_evolution")

        # 3. Extract genes from report (rule‑based)
        raw_genes = self.extractor.extract(report)

        # 4. NEW: Get LLM‑suggested genes (if LLM is available)
        llm_genes = []
        if self.llm:
            try:
                # Run async method in a sync context (since evolve is sync)
                import asyncio
                loop = asyncio.new_event_loop()
                llm_genes = loop.run_until_complete(self._llm_suggest_genes(report))
                loop.close()
            except Exception as e:
                print(f"LLM gene retrieval failed: {e}")

        # Merge and deduplicate by name (LLM genes take precedence if same name)
        all_genes = {g.name: g for g in raw_genes}
        for g in llm_genes:
            if g.name not in all_genes:
                all_genes[g.name] = g
            else:
                # If same name, keep the one with higher confidence
                if g.confidence > all_genes[g.name].confidence:
                    all_genes[g.name] = g
        merged_genes = list(all_genes.values())

        if not merged_genes:
            return current_dna

        # 5. Select best genes
        selected_genes = self.operators.select(merged_genes, top_k=5, min_confidence=0.5)

        # 6. Check if we are in a stable trend (no improvement)
        is_stable = report.overall_trend == "stable →"
        if is_stable and report.accuracy is not None and report.accuracy < 0.7:
            for i, gene in enumerate(selected_genes):
                selected_genes[i] = self.operators.mutate(gene, mutation_rate=0.3, stable_trend=True)

        # 7. Optionally crossover two best genes
        if len(selected_genes) >= 2 and random.random() < 0.2:
            new_gene = self.operators.crossover(selected_genes[0], selected_genes[1])
            if new_gene:
                selected_genes.append(new_gene)

        # 8. Predict impact
        for gene in selected_genes:
            impact = self.predictor.predict(gene, [])
            if not impact.get("recommended", True):
                gene.confidence *= 0.9

        # 9. Consolidate into DNA
        new_dna = self.memory.consolidate(selected_genes, current_dna)

        # 10. Save snapshot after evolution
        self.memory.take_snapshot(new_dna, f"after_evolution_{report.episode_id[:8]}")

        # 11. Record evolution event
        self.evolution_history.append({
            "episode_id": report.episode_id,
            "snapshot_id": snapshot_id,
            "genes_added": [g.name for g in selected_genes],
            "llm_genes_added": [g.name for g in llm_genes],
            "new_version": new_dna.get("version"),
            "timestamp": datetime.now().isoformat(),
        })

        # 12. Save the updated DNA
        self.memory.save_dna(new_dna)

        return new_dna

    def rollback(self, snapshot_id: str) -> bool:
        import glob
        matches = list(self.memory.snapshot_dir.glob(f"{snapshot_id}_*.yaml"))
        if not matches:
            return False
        with open(matches[0], 'r') as f:
            dna_data = yaml.safe_load(f)
        self.memory.save_dna(dna_data)
        return True

    def get_evolution_summary(self) -> str:
        if not self.evolution_history:
            return "No evolution steps recorded yet."
        lines = [f"Evolution steps: {len(self.evolution_history)}"]
        for ev in self.evolution_history[-5:]:
            llm_info = f" (LLM: {len(ev.get('llm_genes_added', []))})" if ev.get('llm_genes_added') else ""
            lines.append(f"  {ev['timestamp'][:19]} | {ev['episode_id'][:8]} | version {ev['new_version']} | +{len(ev['genes_added'])} genes{llm_info}")
        return "\n".join(lines)