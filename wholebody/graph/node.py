from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union


@dataclass
class TensorShape:
    """Tensor shape descriptor for shape propagation and connection validation."""
    dims: Tuple[Optional[Union[int, str]], ...]

    def __repr__(self) -> str:
        return f"({', '.join(str(d) for d in self.dims)})"


@dataclass
class Port:
    """Input or output connection port of a graph node."""
    name: str
    data_type: str = "tensor"
    shape: Optional[Tuple[Any, ...]] = None


@dataclass
class Node:
    """Intermediate Representation (IR) Node representing a neural network module or operation."""
    id: str
    category: str  # 'input', 'backbone', 'neck', 'head', 'loss', 'codec', 'output'
    registry_name: str  # e.g., 'SimpleCNN', 'HeatmapHead', 'MSRAHeatmapCodec'
    params: Dict[str, Any] = field(default_factory=dict)
    input_ports: List[Port] = field(default_factory=list)
    output_ports: List[Port] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "category": self.category,
            "registry_name": self.registry_name,
            "params": self.params,
            "input_ports": [{"name": p.name, "shape": list(p.shape) if p.shape else None} for p in self.input_ports],
            "output_ports": [{"name": p.name, "shape": list(p.shape) if p.shape else None} for p in self.output_ports],
            "metadata": self.metadata,
        }


@dataclass
class Edge:
    """Connection between an output port of a source node and an input port of a target node."""
    source_node: str
    source_port: str
    target_node: str
    target_port: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_node": self.source_node,
            "source_port": self.source_port,
            "target_node": self.target_node,
            "target_port": self.target_port,
        }
