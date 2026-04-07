#!/usr/bin/env python3
"""
Advanced Self‑Improvement Demo – Multi‑Generation Evolution Tracking

Usage:
    python scripts/self_improvement_advanced.py [--generations N] [--real] [--verbose]

Features:
- Runs N generations of a similar ML task
- Tracks accuracy, hallucination, reasoning, cost, duration per generation
- Logs DNA gene additions and confidence changes
- Shows plan evolution (layers, hidden size, learning rate, batch size)
- Exports metrics to experiments/metrics/evolution.csv
- Prints improvement tables and ASCII trend graph
- Uses simulation by default; add --real for actual PyTorch training (requires GPU)
"""

import sys
import time
import csv
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.agent import PrimeAgent, AgentConfig
from src.experience.episodes import EpisodeStore
from src.training.plan import ArchitectureConfig, TrainingConfig


def parse_args():
    parser = argparse.ArgumentParser(description="Advanced self‑improvement demo")
    parser.add_argument("--generations", "-g", type=int, default=5,
                        help="Number of training generations (default: 5)")
    parser.add_argument("--real", action="store_true",
                        help="Use real PyTorch training (requires GPU)")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Print detailed logs")
    return parser.parse_args()


class GenerationTracker:
    """Tracks metrics and changes across generations."""
    def __init__(self):
        self.generations: List[Dict[str, Any]] = []
        self.dna_snapshots: List[Dict] = []
        self.plan_snapshots: List[Dict] = []

    def add_generation(self, gen_num: int, episode, report, duration: float):
        """Store metrics from one generation."""
        ev = episode.evaluation
        metrics = {
            "generation": gen_num,
            "timestamp": datetime.now().isoformat(),
            "status": episode.status.value,
            "accuracy": ev.accuracy if ev else 0.0,
            "hallucination": ev.hallucination_rate if ev else 0.0,
            "reasoning": ev.reasoning_quality if ev else 0.0,
            "safety": ev.safety_score if ev else 0.0,
            "plan_efficiency": ev.plan_efficiency if ev else 0.0,
            "duration_seconds": duration,
            "cost_estimate": ev.token_cost_estimate if ev else 0.0,
            "next_improvement": episode.next_improvement,
            "gene_hints": ", ".join(episode.gene_hints[:3]) if episode.gene_hints else "",
        }
        self.generations.append(metrics)

    def capture_dna(self, dna):
        """Capture a copy of current DNA genes."""
        genes = []
        for g in dna.genes:
            genes.append({
                "name": g.name,
                "type": g.gene_type.value,
                "confidence": g.confidence,
                "value": str(g.value)[:50],
            })
        self.dna_snapshots.append({"generation": len(self.generations), "genes": genes})

    def capture_plan(self, plan):
        """Capture current training plan architecture and hyperparams."""
        if plan and plan.architecture and plan.training:
            self.plan_snapshots.append({
                "generation": len(self.generations),
                "num_layers": plan.architecture.num_layers,
                "hidden_size": plan.architecture.hidden_size,
                "num_heads": plan.architecture.num_heads,
                "learning_rate": plan.training.learning_rate,
                "batch_size": plan.training.batch_size,
                "epochs": plan.training.epochs,
            })

    def print_summary(self):
        """Print a human‑readable evolution summary table."""
        if not self.generations:
            print("No generations recorded.")
            return

        print("\n" + "=" * 80)
        print("EVOLUTION SUMMARY")
        print("=" * 80)

        # Header
        print(f"{'Gen':>4} | {'Accuracy':>8} | {'Hallu':>7} | {'Reasoning':>9} | {'Efficiency':>10} | {'Duration(s)':>11}")
        print("-" * 80)

        # Rows
        for g in self.generations:
            print(f"{g['generation']:4d} | {g['accuracy']:8.3f} | {g['hallucination']:7.3f} | "
                  f"{g['reasoning']:9.3f} | {g['plan_efficiency']:10.2f} | {g['duration_seconds']:11.2f}")

        # Trend detection
        acc_vals = [g['accuracy'] for g in self.generations if g['accuracy'] > 0]
        if len(acc_vals) >= 2:
            delta = acc_vals[-1] - acc_vals[0]
            trend = "improving ↑" if delta > 0 else "declining ↓" if delta < 0 else "stable →"
            print(f"\nAccuracy trend: {acc_vals[0]:.3f} → {acc_vals[-1]:.3f} ({delta:+.3f}) {trend}")

        # ASCII graph of accuracy
        if len(acc_vals) > 1:
            print("\nAccuracy progression (ASCII):")
            max_acc = max(acc_vals)
            min_acc = min(acc_vals)
            if max_acc > min_acc:
                for i, acc in enumerate(acc_vals):
                    bar_len = int((acc - min_acc) / (max_acc - min_acc) * 40)
                    bar = "#" * bar_len
                    print(f"  Gen {i+1:2d}: {acc:.3f} |{bar}")
            else:
                for i, acc in enumerate(acc_vals):
                    print(f"  Gen {i+1:2d}: {acc:.3f} | (no change)")

        # Show plan changes
        if len(self.plan_snapshots) >= 2:
            print("\nPlan evolution (first vs last):")
            first = self.plan_snapshots[0]
            last = self.plan_snapshots[-1]
            print(f"  Layers      : {first['num_layers']} → {last['num_layers']} ({last['num_layers']-first['num_layers']:+d})")
            print(f"  Hidden size : {first['hidden_size']} → {last['hidden_size']} ({last['hidden_size']-first['hidden_size']:+d})")
            print(f"  Learning rate: {first['learning_rate']:.2e} → {last['learning_rate']:.2e}")
            print(f"  Batch size  : {first['batch_size']} → {last['batch_size']} ({last['batch_size']-first['batch_size']:+d})")

        # DNA evolution
        if len(self.dna_snapshots) >= 2:
            first_genes = {g['name']: g['confidence'] for g in self.dna_snapshots[0]['genes']}
            last_genes = {g['name']: g['confidence'] for g in self.dna_snapshots[-1]['genes']}
            added = set(last_genes.keys()) - set(first_genes.keys())
            removed = set(first_genes.keys()) - set(last_genes.keys())
            if added:
                print(f"\nGenes added: {', '.join(added)}")
            if removed:
                print(f"Genes removed: {', '.join(removed)}")
            # Confidence changes
            common = set(first_genes.keys()) & set(last_genes.keys())
            changes = [(g, last_genes[g] - first_genes[g]) for g in common if abs(last_genes[g] - first_genes[g]) > 0.05]
            if changes:
                print("Confidence changes:")
                for g, delta in changes[:5]:
                    print(f"  {g}: {first_genes[g]:.2f} → {last_genes[g]:.2f} ({delta:+.2f})")

    def export_csv(self, filepath: Path):
        """Export metrics to CSV."""
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            if not self.generations:
                return
            writer = csv.DictWriter(f, fieldnames=self.generations[0].keys())
            writer.writeheader()
            writer.writerows(self.generations)
        print(f"\nMetrics exported to: {filepath}")


def main():
    args = parse_args()

    print("\n" + "=" * 80)
    print("ADVANCED SELF‑IMPROVEMENT DEMO")
    print(f"Generations: {args.generations} | Real training: {args.real}")
    print("=" * 80)

    # Initialize agent
    config = AgentConfig(verbose=args.verbose, auto_save=True)
    agent = PrimeAgent(config=config)

    # Optional: clear previous episodes for clean slate (comment out to keep history)
    # store = EpisodeStore(Path("experiments/episodes"))
    # store.prune(keep_recent=0, keep_successful=False, dry_run=False)

    tracker = GenerationTracker()

    # Base task description (same across generations to measure evolution)
    base_task = "Train a small transformer model for step‑by‑step reasoning"

    for gen in range(1, args.generations + 1):
        print(f"\n--- GENERATION {gen} ---")
        start_time = time.time()

        # Run the agent
        result = agent.run(base_task)

        elapsed = time.time() - start_time

        # Get the latest episode
        store = EpisodeStore(Path("experiments/episodes"))
        episodes = store.search(limit=1)
        if not episodes:
            print("  Error: No episode found.")
            continue
        latest_ep = episodes[0]

        # Get the last plan from agent (if available)
        current_plan = agent.current_plan

        # Capture data
        tracker.add_generation(gen, latest_ep, result, elapsed)
        if current_plan:
            tracker.capture_plan(current_plan)
        tracker.capture_dna(agent.dna)

        # Print short summary for this generation
        ev = latest_ep.evaluation
        print(f"  Accuracy: {ev.accuracy if ev else 0:.3f} | Hallucination: {ev.hallucination_rate if ev else 0:.3f}")
        print(f"  Next improvement: {latest_ep.next_improvement[:80] if latest_ep.next_improvement else 'None'}")
        print(f"  Duration: {elapsed:.2f}s")

        # Small pause between generations to let user observe
        time.sleep(1)

    # Final summary
    tracker.print_summary()

    # Export metrics
    csv_path = Path("experiments/metrics/evolution.csv")
    tracker.export_csv(csv_path)

    # Final stats from agent
    print("\n" + "=" * 80)
    print("FINAL AGENT STATISTICS")
    print(agent.stats())
    print("=" * 80)

    print("\nDemo completed. The agent has evolved across multiple generations.")
    print("Check experiments/episodes/ for detailed episode JSON files.")
    print("Check experiments/dna_snapshots/ for DNA version history.")
    print("Check experiments/metrics/evolution.csv for exported metrics.")


if __name__ == "__main__":
    main()