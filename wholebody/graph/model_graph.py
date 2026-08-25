from collections import defaultdict, deque
from typing import Any, Dict, List, Optional, Set, Tuple

from wholebody.graph.node import Edge, Node, Port


class ModelGraph:
    """Directed Acyclic Graph (DAG) intermediate representation of Whole-Body models.
    
    Serves as the bridge between future Visual GUI Builders and PyTorch configuration:
      - Validates topological soundness and tensor shapes
      - Exports directly to valid YAML configs for Registry.build()
      - Imports existing YAML configs into visualizable graphs
    """

    def __init__(self, name: str = "PoseModelGraph") -> None:
        self.name = name
        self.nodes: Dict[str, Node] = {}
        self.edges: List[Edge] = []

    def add_node(self, node: Node) -> None:
        if node.id in self.nodes:
            raise KeyError(f"Node '{node.id}' already exists in graph '{self.name}'")
        self.nodes[node.id] = node

    def add_edge(self, edge: Edge) -> None:
        if edge.source_node not in self.nodes:
            raise KeyError(f"Source node '{edge.source_node}' not found in graph.")
        if edge.target_node not in self.nodes:
            raise KeyError(f"Target node '{edge.target_node}' not found in graph.")
        self.edges.append(edge)

    def validate_topology(self) -> List[str]:
        """Verify graph is a valid DAG and return topological order of node IDs."""
        in_degree: Dict[str, int] = {node_id: 0 for node_id in self.nodes}
        adj_list: Dict[str, List[str]] = defaultdict(list)

        for edge in self.edges:
            adj_list[edge.source_node].append(edge.target_node)
            in_degree[edge.target_node] += 1

        queue = deque([node_id for node_id, deg in in_degree.items() if deg == 0])
        order: List[str] = []

        while queue:
            curr = queue.popleft()
            order.append(curr)
            for neighbor in adj_list[curr]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        if len(order) != len(self.nodes):
            raise ValueError(f"Cycle detected in ModelGraph '{self.name}'. Graph must be a DAG.")

        return order

    def to_config(self) -> Dict[str, Any]:
        """Convert graph representation into a standard WholeBody model configuration dict."""
        self.validate_topology()

        # Find backbone, neck, head nodes
        backbone_node = next((n for n in self.nodes.values() if n.category == "backbone"), None)
        neck_node = next((n for n in self.nodes.values() if n.category == "neck"), None)
        head_node = next((n for n in self.nodes.values() if n.category == "head"), None)

        if backbone_node is None or head_node is None:
            raise ValueError("ModelGraph must contain at least one 'backbone' and one 'head' node.")

        model_cfg: Dict[str, Any] = {
            "type": "TopDownPoseEstimator",
            "backbone": {
                "type": backbone_node.registry_name,
                **backbone_node.params,
            },
            "head": {
                "type": head_node.registry_name,
                **head_node.params,
            },
        }

        if neck_node is not None:
            model_cfg["neck"] = {
                "type": neck_node.registry_name,
                **neck_node.params,
            }

        return {"model": model_cfg}

    @classmethod
    def from_config(cls, cfg: Dict[str, Any], name: str = "PoseModelGraph") -> "ModelGraph":
        """Construct ModelGraph IR from a standard model configuration dictionary."""
        graph = cls(name=name)
        model_dict = cfg.get("model", cfg)

        # Input Node
        in_node = Node(
            id="input",
            category="input",
            registry_name="InputTensor",
            output_ports=[Port(name="out", shape=(None, 3, 256, 192))],
        )
        graph.add_node(in_node)

        # Backbone Node
        bb_cfg = dict(model_dict["backbone"])
        bb_type = bb_cfg.pop("type")
        bb_node = Node(
            id="backbone",
            category="backbone",
            registry_name=bb_type,
            params=bb_cfg,
            input_ports=[Port(name="in")],
            output_ports=[Port(name="out")],
        )
        graph.add_node(bb_node)
        graph.add_edge(Edge(source_node="input", source_port="out", target_node="backbone", target_port="in"))

        prev_node_id = "backbone"

        # Optional Neck
        if "neck" in model_dict and model_dict["neck"] is not None:
            neck_cfg = dict(model_dict["neck"])
            neck_type = neck_cfg.pop("type")
            neck_node = Node(
                id="neck",
                category="neck",
                registry_name=neck_type,
                params=neck_cfg,
                input_ports=[Port(name="in")],
                output_ports=[Port(name="out")],
            )
            graph.add_node(neck_node)
            graph.add_edge(Edge(source_node="backbone", source_port="out", target_node="neck", target_port="in"))
            prev_node_id = "neck"

        # Head Node
        head_cfg = dict(model_dict["head"])
        head_type = head_cfg.pop("type")
        head_node = Node(
            id="head",
            category="head",
            registry_name=head_type,
            params=head_cfg,
            input_ports=[Port(name="in")],
            output_ports=[Port(name="out")],
        )
        graph.add_node(head_node)
        graph.add_edge(Edge(source_node=prev_node_id, source_port="out", target_node="head", target_port="in"))

        return graph
