#!/usr/bin/env python3
"""Visualize the golden test graph as a PNG.

Reads the golden graph definition (NODES, EDGES) from the test fixtures and
produces a ForceAtlas2 force-directed visualization with color-coded node
types and edge predicates.  Symmetric predicates are drawn as single directed
arrows.

Usage:
    uv run python scripts/visualize_golden_graph.py
    uv run python scripts/visualize_golden_graph.py --output path/to/out.png
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import networkx as nx

sys.path.insert(0, str(Path(__file__).parent.parent / "tests" / "test_end_to_end"))
from golden_graph import EDGES, GRAPH_STATS, NODES

# ---------------------------------------------------------------------------
# Visual style
# ---------------------------------------------------------------------------

NODE_COLORS = {
    "Gene":          "#66bb6a",   # green
    "Protein":       "#4dd0e1",   # cyan
    "Disease":       "#f48fb1",   # pink
    "SmallMolecule": "#ce93d8",   # purple
    "Gene+Protein":  "#fff176",   # yellow  (pseudo-type)
}

PRED_COLORS = {
    "affects":         "#e53935",  # red
    "treats":          "#00897b",  # teal
    "interacts_with":  "#1565c0",  # dark blue
    "regulates":       "#e65100",  # orange
    "associated_with": "#43a047",  # dark green
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _node_display_type(node_id: str) -> str:
    for node in NODES:
        if node["id"] == node_id:
            cats = [c.replace("biolink:", "") for c in node["category"]]
            if "Gene" in cats and "Protein" in cats:
                return "Gene+Protein"
            return cats[0]
    return "Unknown"


def _short_label(node_id: str) -> str:
    return node_id.replace("TEST:", "")


def _build_digraph() -> nx.MultiDiGraph:
    """Directed multigraph used for drawing."""
    G = nx.MultiDiGraph()
    for node in NODES:
        nid = node["id"]
        G.add_node(nid, label=_short_label(nid), ntype=_node_display_type(nid))
    for edge in EDGES:
        pred = edge["predicate"].replace("biolink:", "")
        G.add_edge(edge["subject"], edge["object"], pred=pred)
    return G


def _layout_graph(G: nx.MultiDiGraph, seed: int = 42) -> dict:
    """Compute ForceAtlas2 positions on a simple undirected view of the graph.

    Using a plain undirected graph (one edge per node pair regardless of edge
    count or direction) avoids position distortions from multi-edges.
    """
    U = nx.Graph()
    U.add_nodes_from(G.nodes())
    for u, v in G.edges():
        if u != v:
            U.add_edge(u, v)
    return nx.forceatlas2_layout(U, seed=seed)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    default_out = str(
        Path(__file__).parent.parent / "tests" / "test_end_to_end" / "golden_graph.png"
    )
    parser = argparse.ArgumentParser(description="Visualize the golden test graph")
    parser.add_argument("--output", default=default_out, help="Output PNG path")
    parser.add_argument("--seed", type=int, default=42, help="Spring layout random seed")
    args = parser.parse_args()

    G = _build_digraph()
    pos = _layout_graph(G, seed=args.seed)

    fig, ax = plt.subplots(figsize=(18, 13))
    ax.set_facecolor("#fafafa")

    # --- nodes ---
    for ntype, color in NODE_COLORS.items():
        nodelist = [n for n in G.nodes() if G.nodes[n]["ntype"] == ntype]
        if nodelist:
            nx.draw_networkx_nodes(
                G, pos, ax=ax,
                nodelist=nodelist,
                node_color=color,
                node_size=2400,
                alpha=0.95,
                linewidths=1.5,
                edgecolors="#555",
            )

    nx.draw_networkx_labels(
        G, pos,
        labels={n: G.nodes[n]["label"] for n in G.nodes()},
        ax=ax,
        font_size=9,
        font_weight="bold",
    )

    # --- edges ---
    margin = 30

    for pred, color in PRED_COLORS.items():
        edgelist = [(u, v) for u, v, d in G.edges(data=True) if d["pred"] == pred]
        if not edgelist:
            continue
        nx.draw_networkx_edges(
            G, pos, edgelist=edgelist, ax=ax,
            edge_color=color, width=2.0,
            arrowsize=22, arrowstyle="->",
            min_source_margin=margin,
            min_target_margin=margin,
        )

    # --- edge labels ---
    edge_label_map: dict[tuple, str] = {}
    for u, v, d in G.edges(data=True):
        key = (u, v)
        p = d["pred"]
        if key not in edge_label_map:
            edge_label_map[key] = p
        elif p not in edge_label_map[key].split("\n"):
            edge_label_map[key] += f"\n{p}"

    nx.draw_networkx_edge_labels(
        G, pos,
        edge_labels=edge_label_map,
        ax=ax,
        font_size=7,
        label_pos=0.38,
        bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.75, ec="none"),
    )

    # --- legend ---
    node_handles = [
        mpatches.Patch(facecolor=c, edgecolor="#555", label=t)
        for t, c in NODE_COLORS.items()
    ]
    pred_handles = [
        mpatches.Patch(color=c, label=p)
        for p, c in PRED_COLORS.items()
    ]
    ax.legend(
        handles=node_handles + pred_handles,
        loc="upper left",
        fontsize=9,
        ncol=2,
        title="Node types  /  Edge predicates",
        title_fontsize=9,
        framealpha=0.9,
    )

    n_nodes = GRAPH_STATS["num_nodes"]
    n_edges = len(EDGES)
    ax.set_title(
        f"Golden Test Graph  ·  {n_nodes} nodes  ·  {n_edges} edges  ·  "
        "hierarchical aggregation, pseudo-types, symmetric predicates, 2-hop triangles",
        fontsize=12,
        pad=14,
    )
    ax.axis("off")

    plt.tight_layout()
    plt.savefig(args.output, dpi=150, bbox_inches="tight")
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
