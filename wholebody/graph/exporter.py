import json
from pathlib import Path
from typing import Any, Dict, Optional, Union

from wholebody.graph.model_graph import ModelGraph


def to_json(graph: ModelGraph, filepath: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
    """Export ModelGraph to JSON schema compatible with React Flow / Web GUI Builders."""
    data = {
        "name": graph.name,
        "nodes": [node.to_dict() for node in graph.nodes.values()],
        "edges": [edge.to_dict() for edge in graph.edges],
    }
    if filepath is not None:
        p = Path(filepath)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
    return data


def export_dot(graph: ModelGraph, filepath: Optional[Union[str, Path]] = None) -> str:
    """Generate Graphviz DOT representation for architecture diagrams."""
    lines = [f'digraph "{graph.name}" {{', "  rankdir=LR;", "  node [shape=box, style=rounded];"]

    category_colors = {
        "input": "#E1BEE7",
        "backbone": "#BBDEFB",
        "neck": "#C8E6C9",
        "head": "#FFE0B2",
        "loss": "#FFCDD2",
        "codec": "#FFF9C4",
        "output": "#D1C4E9",
    }

    for node in graph.nodes.values():
        color = category_colors.get(node.category, "#FFFFFF")
        label = f"{node.category.upper()}\\n{node.registry_name}"
        lines.append(f'  "{node.id}" [label="{label}", fillcolor="{color}", style="filled,rounded"];')

    for edge in graph.edges:
        lines.append(f'  "{edge.source_node}" -> "{edge.target_node}" [label="{edge.source_port}->{edge.target_port}"];')

    lines.append("}")
    dot_str = "\n".join(lines)

    if filepath is not None:
        p = Path(filepath)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            f.write(dot_str)

    return dot_str
