"""
Graphviz backend for MathExpr graphs.
"""

# Imports
from __future__ import annotations
from typing import Dict
from graphviz import Digraph
from .extract import Graph


INTERNAL_NODE_STYLE = {
    "shape": "box",
    "style": "rounded,filled",
    "fillcolor": "#EDF2F7",
}

LEAF_NODE_STYLE = {
    "shape": "box",
    "style": "filled",
    "fillcolor": "#E6FFFA",
}


def render(
        graph: Graph,
        **kwargs
) -> Digraph:
    """
    Render the graph using Graphviz Digraph.

    Parameters
    ----------
    graph : Graph
        Graph to render.
    """
    dot = Digraph(
        engine='dot',
        graph_attr={
            "rankdir": "TB",
            "nodesep": "0.8",
            "ranksep": "1.2",
        },
        **kwargs
    )

    # Add nodes
    for node in graph.nodes.values():
        if node.meta['is_node']:
            dot.attr("node", **INTERNAL_NODE_STYLE)
            node_label = (
                f"{node.meta['id']}: {node.meta['op']} = {node.meta['eval']}\\n"
                f"{node.meta['name']} ({node.meta['arity']})\\n"
                f"{node.meta['dtype']}\\n"
                f"{node.meta['shape']}"
            )
        else:
            dot.attr("node", **LEAF_NODE_STYLE)
            leaf_type = "Variable" if node.meta['mutable'] else "Constant"
            node_label = (f"{leaf_type} {node.meta['name']}={node.meta['eval']}\\n"
                          f"({node.meta['dtype']})"
                          f"\\n{node.meta['shape']}")
        # end if
        dot.node(node.id, label=node_label)
    # end for

    # Add edges
    dot.attr("edge", color="#2D3748")
    for parent, child in graph.edges:
        dot.edge(parent, child)
    # end for

    return dot
# end def render