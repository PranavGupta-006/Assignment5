"""
Q4: Knowledge Base Graphs
Author: Pranav Gupta
Roll No: SE24UCSE020
College: Mahindra University

Tech stack: Python, networkx, matplotlib, pyvis, pandas, json, sys, time
This file builds a small knowledge graph, computes graph metrics, and exports
static plus interactive visualizations for the Indian tech ecosystem example.
"""

import json
import os
import sys
import time

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
from pyvis.network import Network


# ── Output directory ──────────────────────────────────────────────────────────
OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ── Color palette by entity type ──────────────────────────────────────────────
COLOR_MAP = {
    "Startup": "#2ECC71",
    "Investor": "#E74C3C",
    "Product": "#3498DB",
    "Founder": "#F39C12",
}
DEFAULT_COLOR = "#95A5A6"


# ─────────────────────────────────────────────────────────────────────────────
# 1. Data definitions
# ─────────────────────────────────────────────────────────────────────────────

def load_data():
    """Return entity and relationship DataFrames for an Indian tech ecosystem."""

    nodes = pd.DataFrame([
        {"name": "Zepto", "type": "Startup"},
        {"name": "Razorpay", "type": "Startup"},
        {"name": "Meesho", "type": "Startup"},
        {"name": "Sequoia", "type": "Investor"},
        {"name": "Y Combinator", "type": "Investor"},
        {"name": "Aadit Palicha", "type": "Founder"},
        {"name": "Harshil Mathur", "type": "Founder"},
        {"name": "FastPay", "type": "Product"},
        {"name": "RazorpayX", "type": "Product"},
    ])

    edges = pd.DataFrame([
        {"source": "Aadit Palicha", "target": "Zepto", "rel": "founded"},
        {"source": "Harshil Mathur", "target": "Razorpay", "rel": "founded"},
        {"source": "Sequoia", "target": "Zepto", "rel": "invested_in"},
        {"source": "Sequoia", "target": "Meesho", "rel": "invested_in"},
        {"source": "Y Combinator", "target": "Razorpay", "rel": "invested_in"},
        {"source": "Y Combinator", "target": "Meesho", "rel": "invested_in"},
        {"source": "Razorpay", "target": "FastPay", "rel": "launched"},
        {"source": "Razorpay", "target": "RazorpayX", "rel": "launched"},
        {"source": "Zepto", "target": "Razorpay", "rel": "partnered_with"},
    ])

    return nodes, edges


# ─────────────────────────────────────────────────────────────────────────────
# 2. Graph construction
# ─────────────────────────────────────────────────────────────────────────────

def build_graph(nodes: pd.DataFrame, edges: pd.DataFrame) -> nx.DiGraph:
    """Construct a directed knowledge graph from node and edge DataFrames."""

    g = nx.DiGraph()

    for _, row in nodes.iterrows():
        g.add_node(row["name"], type=row["type"])

    for _, row in edges.iterrows():
        g.add_edge(row["source"], row["target"], relationship=row["rel"])

    return g


def graph_edge_payload(g: nx.DiGraph):
    """Convert graph edges into a frontend-friendly payload."""

    return [
        {"source": source, "target": target, "relation": attrs["relationship"]}
        for source, target, attrs in g.edges(data=True)
    ]


def graph_tool_catalog():
    """Return graph tooling categories that mirror the other assignment pages."""

    return [
        {
            "category": "Graph databases",
            "tools": ["Neo4j", "Memgraph", "Amazon Neptune", "Azure Cosmos DB", "AllegroGraph", "GraphDB", "JanusGraph"],
            "use_case": "Store and query highly connected startup, investor, founder, and product data.",
        },
        {
            "category": "Graph construction",
            "tools": ["LlamaIndex", "LangChain", "GliNER", "Infernotus", "ContextClue Graph Builder"],
            "use_case": "Extract entities and relationships from notes, reports, profiles, and tables.",
        },
        {
            "category": "Ontology modeling",
            "tools": ["Protege", "TopBraid Composer"],
            "use_case": "Define reusable classes, properties, and constraints for graph data.",
        },
        {
            "category": "Visualization",
            "tools": ["Gephi", "Kumu", "Linkurious"],
            "use_case": "Explore hubs, communities, and partner paths in the ecosystem.",
        },
        {
            "category": "Personal KGs",
            "tools": ["Obsidian Graph View", "TheBrain"],
            "use_case": "Model a founder's notes, contacts, and company relationships.",
        },
    ]


# ─────────────────────────────────────────────────────────────────────────────
# 3. Console summary
# ─────────────────────────────────────────────────────────────────────────────

def print_summary(g: nx.DiGraph):
    """Print graph stats, centrality scores, and shortest-path examples."""

    print("\nKnowledge Graph — Tech Ecosystem")
    print("=" * 45)
    print(f"  Nodes : {g.number_of_nodes()}")
    print(f"  Edges : {g.number_of_edges()}")

    print("\nEntities:")
    for node, attrs in g.nodes(data=True):
        print(f"  [{attrs.get('type', '?')}]  {node}")

    print("\nRelationships:")
    for src, tgt, attrs in g.edges(data=True):
        print(f"  {src}  ──{attrs['relationship']}──▶  {tgt}")

    # ── Degree centrality ────────────────────────────────────────────────────
    print("\nDegree Centrality (higher = more connected)")
    print("-" * 45)
    deg_centrality = nx.degree_centrality(g)
    for node, score in sorted(deg_centrality.items(), key=lambda x: -x[1]):
        print(f"  {node:<22} {score:.3f}")

    # ── Betweenness centrality ───────────────────────────────────────────────
    print("\nBetweenness Centrality (higher = more of a bridge)")
    print("-" * 45)
    between = nx.betweenness_centrality(g)
    for node, score in sorted(between.items(), key=lambda x: -x[1]):
        print(f"  {node:<22} {score:.3f}")

    # ── Shortest path example ────────────────────────────────────────────────
    print("\nShortest Path Examples")
    print("-" * 45)
    undirected = g.to_undirected()
    pairs = [
        ("Aadit Palicha", "Y Combinator"),
        ("Sequoia", "RazorpayX"),
    ]
    for start, end in pairs:
        try:
            path = nx.shortest_path(undirected, source=start, target=end)
            print(f"  {start} → {end}:  {' → '.join(path)}")
        except nx.NetworkXNoPath:
            print(f"  {start} → {end}:  no path found")


# ─────────────────────────────────────────────────────────────────────────────
# 4. Static matplotlib visualization
# ─────────────────────────────────────────────────────────────────────────────

def save_static_plot(g: nx.DiGraph, filepath: str):
    """Render and save a static PNG of the knowledge graph."""

    node_colors = [
        COLOR_MAP.get(g.nodes[n].get("type", ""), DEFAULT_COLOR)
        for n in g.nodes()
    ]

    fig, ax = plt.subplots(figsize=(12, 8))
    pos = nx.spring_layout(g, seed=7, k=1.8)

    nx.draw(
        g,
        pos,
        ax=ax,
        with_labels=True,
        node_color=node_colors,
        node_size=3200,
        font_size=9,
        font_color="white",
        font_weight="bold",
        arrows=True,
        arrowsize=18,
        edge_color="#555555",
    )

    edge_labels = {
        (u, v): attrs["relationship"]
        for u, v, attrs in g.edges(data=True)
    }
    nx.draw_networkx_edge_labels(
        g,
        pos,
        edge_labels=edge_labels,
        font_size=8,
        ax=ax,
    )

    legend_patches = [
        mpatches.Patch(color=color, label=entity_type)
        for entity_type, color in COLOR_MAP.items()
    ]
    ax.legend(handles=legend_patches, loc="upper left", fontsize=9)
    ax.set_title("Indian Tech Ecosystem — Knowledge Graph", fontsize=13, pad=14)

    plt.tight_layout()
    plt.savefig(filepath, dpi=150)
    plt.close()
    print(f"\nStatic visualization saved → {filepath}")


# ─────────────────────────────────────────────────────────────────────────────
# 5. Interactive PyVis visualization
# ─────────────────────────────────────────────────────────────────────────────

def save_interactive_graph(g: nx.DiGraph, filepath: str):
    """Generate a standalone interactive HTML graph using PyVis."""

    net = Network(
        height="750px",
        width="100%",
        directed=True,
        notebook=False,
        bgcolor="#1a1a2e",
        font_color="white",
    )

    for node, attrs in g.nodes(data=True):
        entity_type = attrs.get("type", "Unknown")
        net.add_node(
            node,
            label=node,
            title=f"Type: {entity_type}",
            color=COLOR_MAP.get(entity_type, DEFAULT_COLOR),
            size=22,
        )

    for src, tgt, attrs in g.edges(data=True):
        net.add_edge(
            src,
            tgt,
            title=attrs["relationship"],
            label=attrs["relationship"],
            color="#aaaaaa",
        )

    net.write_html(filepath)
    print(f"Interactive graph saved → {filepath}")


# ─────────────────────────────────────────────────────────────────────────────
# 6. Export helpers
# ─────────────────────────────────────────────────────────────────────────────

def export_graphml(g: nx.DiGraph, filepath: str):
    """Export the graph as GraphML for external tools."""

    nx.write_graphml(g, filepath)
    print(f"GraphML exported → {filepath}")


def emit_step(title, status="running", detail="", data=None):
    print(json.dumps({
        "title": title,
        "status": status,
        "detail": detail,
        "data": data or {},
    }), flush=True)


def run_console():
    """Run the original console workflow."""

    node_df, edge_df = load_data()
    graph = build_graph(node_df, edge_df)

    print_summary(graph)
    save_static_plot(graph, f"{OUTPUT_DIR}/graph_static.png")
    save_interactive_graph(graph, f"{OUTPUT_DIR}/graph_interactive.html")
    export_graphml(graph, f"{OUTPUT_DIR}/graph.graphml")

    print("\nDone.")


def run_api_steps():
    """Stream step-by-step JSON for the Vite frontend."""

    start = time.perf_counter()
    node_df, edge_df = load_data()
    graph = build_graph(node_df, edge_df)
    edges = graph_edge_payload(graph)

    emit_step(
        "Load graph dataset",
        detail="Preparing startups, investors, founders, products, and their relationships.",
        data={
            "nodes": graph.number_of_nodes(),
            "edges": graph.number_of_edges(),
            "graph_edges": edges,
            "tools": graph_tool_catalog(),
            "visual_type": "knowledge-graph",
        },
    )

    deg_centrality = nx.degree_centrality(graph)
    between = nx.betweenness_centrality(graph)
    emit_step(
        "Analyze graph connectivity",
        detail="Calculating centrality to identify the most connected and influential nodes.",
        data={
            "notes": [
                f"{node}: degree {score:.3f}"
                for node, score in sorted(deg_centrality.items(), key=lambda item: -item[1])[:5]
            ] + [
                f"{node}: betweenness {score:.3f}"
                for node, score in sorted(between.items(), key=lambda item: -item[1])[:5]
            ],
            "visual_type": "summary-notes",
        },
    )

    undirected = graph.to_undirected()
    path_notes = []
    for start_node, end_node in [
        ("Aadit Palicha", "Y Combinator"),
        ("Sequoia", "RazorpayX"),
    ]:
        try:
            path = nx.shortest_path(undirected, source=start_node, target=end_node)
            path_notes.append(f"{start_node} -> {end_node}: {' -> '.join(path)}")
        except nx.NetworkXNoPath:
            path_notes.append(f"{start_node} -> {end_node}: no path found")

    emit_step(
        "Trace relationship paths",
        detail="Showing example connection paths through the knowledge graph.",
        data={
            "notes": path_notes,
            "visual_type": "summary-notes",
        },
    )

    emit_step(
        "Q4 complete",
        status="complete",
        detail="Knowledge-base graph workflow finished successfully.",
        data={"elapsed_ms": round((time.perf_counter() - start) * 1000, 2)},
    )


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if "--steps" in sys.argv:
        run_api_steps()
    else:
        run_console()
