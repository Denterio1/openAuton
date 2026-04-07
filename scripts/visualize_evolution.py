#!/usr/bin/env python3
"""
visualize_evolution.py – DNA Lineage Graph

Reads episodes and DNA snapshots to draw a tree of how the agent evolved.
Output: experiments/lineage_graph/dna_evolution.png

Usage:
    python scripts/visualize_evolution.py
"""

import sys
import json
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.experience.episodes import EpisodeStore

# Optional imports (fail gracefully)
try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import networkx as nx
    HAS_VIS = True
except ImportError:
    HAS_VIS = False
    print("Warning: matplotlib or networkx not installed. Install with: pip install matplotlib networkx")


def load_episodes(store_dir: Path):
    """Load all episodes and build lineage data."""
    store = EpisodeStore(store_dir)
    episodes = store.load_all()
    # Sort by timestamp
    episodes.sort(key=lambda e: e.timestamp)
    return episodes


def build_lineage(episodes):
    """Build parent-child relationships for graph."""
    edges = []  # (parent_id, child_id)
    nodes = []  # (episode_id, accuracy, generation, gene_count, status)
    for ep in episodes:
        acc = ep.evaluation.accuracy if ep.evaluation else 0.0
        gene_count = len(ep.gene_hints) if ep.gene_hints else 0
        nodes.append((ep.episode_id, acc, ep.timestamp, gene_count, ep.status.value))
        if ep.parent_episode_id:
            edges.append((ep.parent_episode_id, ep.episode_id))
    return nodes, edges


def draw_lineage(nodes, edges, output_path: Path):
    """Create and save the graph."""
    if not HAS_VIS:
        print("Visualisation libraries not installed. Skipping graph generation.")
        return

    G = nx.DiGraph()
    for nid, acc, ts, gc, status in nodes:
        # Node label: short ID + accuracy
        label = f"{nid[:6]}\nacc={acc:.2f}"
        G.add_node(nid, label=label, accuracy=acc, timestamp=ts, gene_count=gc, status=status)
    for parent, child in edges:
        G.add_edge(parent, child)

    plt.figure(figsize=(12, 8))
    # Use hierarchical layout (top to bottom)
    pos = nx.nx_agraph.graphviz_layout(G, prog='dot') if nx.nx_agraph else nx.spring_layout(G, k=2, iterations=50)
    # Color nodes by accuracy
    accs = [G.nodes[n].get('accuracy', 0) for n in G.nodes]
    node_colors = plt.cm.RdYlGn([(a - min(accs)) / (max(accs) - min(accs) + 1e-6) for a in accs])
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=800, alpha=0.9)
    nx.draw_networkx_edges(G, pos, edge_color='gray', arrows=True, arrowsize=15, arrowstyle='->')
    nx.draw_networkx_labels(G, pos, labels={n: G.nodes[n]['label'] for n in G.nodes}, font_size=8)
    plt.title("DNA Lineage – Agent Evolution", fontsize=14)
    plt.axis('off')
    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap='RdYlGn', norm=plt.Normalize(vmin=min(accs), vmax=max(accs)))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=plt.gca(), shrink=0.6)
    cbar.set_label('Accuracy')
    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Graph saved to {output_path}")


def main():
    store_dir = Path("experiments/episodes")
    output_dir = Path("experiments/lineage_graph")
    output_path = output_dir / "dna_evolution.png"

    if not store_dir.exists():
        print(f"No episodes found in {store_dir}. Run the agent first.")
        return

    episodes = load_episodes(store_dir)
    if len(episodes) < 2:
        print(f"Only {len(episodes)} episode(s). Need at least 2 to show evolution.")
        return

    nodes, edges = build_lineage(episodes)
    draw_lineage(nodes, edges, output_path)

    # Print summary
    print(f"Lineage built from {len(episodes)} episodes")
    print(f"Edges: {len(edges)} parent-child relationships")
    print(f"Output: {output_path}")


if __name__ == "__main__":
    main()