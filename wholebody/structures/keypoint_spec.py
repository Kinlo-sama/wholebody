from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

from wholebody.core.registry import KEYPOINT_SPECS


@dataclass
class JointSpec:
    """Specification of an individual keypoint joint."""
    id: int
    name: str
    group: str  # e.g., 'body', 'face', 'left_hand', 'right_hand', 'feet'
    color: Tuple[int, int, int] = (0, 255, 0)  # RGB
    weight: float = 1.0  # Loss weighting
    flip_pair: Optional[str] = None  # Counterpart joint name for horizontal flip
    sigma: float = 0.025  # OKS evaluation standard deviation


@dataclass
class SkeletonEdge:
    """Specification of a limb connection between two keypoint joints."""
    joint1_id: int
    joint2_id: int
    color: Tuple[int, int, int] = (255, 0, 0)  # RGB
    thickness: int = 2


@KEYPOINT_SPECS.register("KeypointSpec")
class KeypointSpec:
    """Universal definition for keypoint skeletons and whole-body topologies.
    
    Decouples models, datasets, and visualizers from hardcoded keypoint counts.
    Supports arbitrary keypoint definitions (17, 133, 136, custom).
    """

    def __init__(
        self,
        name: str,
        joints: List[Union[JointSpec, Dict[str, Any]]],
        edges: Optional[List[Union[SkeletonEdge, Tuple[int, int], Dict[str, Any]]]] = None,
    ) -> None:
        self.name = name
        self.joints: Dict[int, JointSpec] = {}
        self.joint_name_to_id: Dict[str, int] = {}
        self.groups: Dict[str, List[int]] = {}

        # Parse joints
        for i, item in enumerate(joints):
            if isinstance(item, dict):
                spec = JointSpec(**item)
            else:
                spec = item
            self.joints[spec.id] = spec
            self.joint_name_to_id[spec.name] = spec.id

            if spec.group not in self.groups:
                self.groups[spec.group] = []
            self.groups[spec.group].append(spec.id)

        self.num_keypoints = len(self.joints)

        # Build flip index mapping
        self.flip_indices = list(range(self.num_keypoints))
        for j_id, j_spec in self.joints.items():
            if j_spec.flip_pair is not None and j_spec.flip_pair in self.joint_name_to_id:
                partner_id = self.joint_name_to_id[j_spec.flip_pair]
                self.flip_indices[j_id] = partner_id

        # Sigmas and weights arrays
        self.sigmas = np.array([self.joints[i].sigma for i in range(self.num_keypoints)], dtype=np.float32)
        self.weights = np.array([self.joints[i].weight for i in range(self.num_keypoints)], dtype=np.float32)

        # Parse skeleton edges
        self.edges: List[SkeletonEdge] = []
        if edges is not None:
            for item in edges:
                if isinstance(item, SkeletonEdge):
                    self.edges.append(item)
                elif isinstance(item, dict):
                    self.edges.append(SkeletonEdge(**item))
                elif isinstance(item, (tuple, list)):
                    self.edges.append(SkeletonEdge(joint1_id=item[0], joint2_id=item[1]))

    def get_group_indices(self, group_name: str) -> List[int]:
        """Return list of joint IDs belonging to a semantic group (e.g. 'face', 'left_hand')."""
        return self.groups.get(group_name, [])

    def to_dict(self) -> Dict[str, Any]:
        """Serialize spec to a standard dictionary."""
        return {
            "name": self.name,
            "joints": [
                {
                    "id": j.id,
                    "name": j.name,
                    "group": j.group,
                    "color": list(j.color),
                    "weight": j.weight,
                    "flip_pair": j.flip_pair,
                    "sigma": j.sigma,
                }
                for j in self.joints.values()
            ],
            "edges": [
                {
                    "joint1_id": e.joint1_id,
                    "joint2_id": e.joint2_id,
                    "color": list(e.color),
                    "thickness": e.thickness,
                }
                for e in self.edges
            ],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "KeypointSpec":
        return cls(name=data["name"], joints=data["joints"], edges=data.get("edges", []))

    def __repr__(self) -> str:
        return f"KeypointSpec(name='{self.name}', num_keypoints={self.num_keypoints}, groups={list(self.groups.keys())})"


# ---------------------------------------------------------------------------
# Standard Presets: COCO-17 and COCO-WholeBody-133
# ---------------------------------------------------------------------------

def create_coco_17_spec() -> KeypointSpec:
    """Create standard COCO 17-keypoint specification."""
    names = [
        "nose", "left_eye", "right_eye", "left_ear", "right_ear",
        "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
        "left_wrist", "right_wrist", "left_hip", "right_hip",
        "left_knee", "right_knee", "left_ankle", "right_ankle"
    ]
    flip_pairs = {
        "left_eye": "right_eye", "right_eye": "left_eye",
        "left_ear": "right_ear", "right_ear": "left_ear",
        "left_shoulder": "right_shoulder", "right_shoulder": "left_shoulder",
        "left_elbow": "right_elbow", "right_elbow": "left_elbow",
        "left_wrist": "right_wrist", "right_wrist": "left_wrist",
        "left_hip": "right_hip", "right_hip": "left_hip",
        "left_knee": "right_knee", "right_knee": "left_knee",
        "left_ankle": "right_ankle", "right_ankle": "left_ankle",
    }
    sigmas = [
        0.026, 0.025, 0.025, 0.035, 0.035,
        0.079, 0.079, 0.072, 0.072,
        0.062, 0.062, 0.107, 0.107,
        0.087, 0.087, 0.089, 0.089
    ]
    joints = [
        JointSpec(
            id=i,
            name=name,
            group="body",
            color=(0, 255, 128) if "left" in name else ((255, 128, 0) if "right" in name else (255, 255, 0)),
            weight=1.0,
            flip_pair=flip_pairs.get(name),
            sigma=sigmas[i]
        )
        for i, name in enumerate(names)
    ]
    edges = [
        (15, 13), (13, 11), (16, 14), (14, 12), (11, 12),
        (5, 11), (6, 12), (5, 6), (5, 7), (6, 8),
        (7, 9), (8, 10), (1, 2), (0, 1), (0, 2),
        (1, 3), (2, 4), (3, 5), (4, 6)
    ]
    return KeypointSpec("coco_17", joints, edges)


def create_coco_wholebody_133_spec() -> KeypointSpec:
    """Create standard COCO-WholeBody 133-keypoint specification."""
    # 0-16: Body (17)
    coco17 = create_coco_17_spec()
    joints = list(coco17.joints.values())
    edges: List[Tuple[int, int]] = [(e.joint1_id, e.joint2_id) for e in coco17.edges]

    # 17-22: Feet (6) -> 3 per foot
    feet_names = [
        "left_big_toe", "left_small_toe", "left_heel",
        "right_big_toe", "right_small_toe", "right_heel"
    ]
    feet_flip = {
        "left_big_toe": "right_big_toe", "right_big_toe": "left_big_toe",
        "left_small_toe": "right_small_toe", "right_small_toe": "left_small_toe",
        "left_heel": "right_heel", "right_heel": "left_heel",
    }
    for i, name in enumerate(feet_names, start=17):
        joints.append(JointSpec(
            id=i, name=name, group="feet",
            color=(0, 200, 255) if "left" in name else (255, 200, 0),
            weight=1.0, flip_pair=feet_flip.get(name), sigma=0.068
        ))
    # Feet edges connecting to ankles (15: left_ankle, 16: right_ankle)
    edges.extend([(15, 17), (17, 18), (18, 19), (16, 20), (20, 21), (21, 22)])

    # 23-90: Face (68)
    for i in range(68):
        j_id = 23 + i
        name = f"face_{i}"
        # 68-face symmetric flipping
        if i < 17:
            flip_name = f"face_{16 - i}"
        elif i < 27:
            flip_name = f"face_{26 - (i - 17)}"
        elif i < 36:
            flip_name = f"face_{i}" if i in (30, 33) else (f"face_{35 - (i - 31)}" if i >= 31 else f"face_{i}")
        elif i < 48:
            flip_name = f"face_{i + 6}" if i < 42 else f"face_{i - 6}"
        else:
            flip_name = f"face_{i}"
        joints.append(JointSpec(
            id=j_id, name=name, group="face",
            color=(255, 100, 200), weight=1.0, flip_pair=flip_name, sigma=0.025
        ))
    # Face outline edges
    for i in range(16):
        edges.append((23 + i, 23 + i + 1))

    # 91-111: Left Hand (21)
    for i in range(21):
        j_id = 91 + i
        name = f"left_hand_{i}"
        flip_name = f"right_hand_{i}"
        joints.append(JointSpec(
            id=j_id, name=name, group="left_hand",
            color=(100, 255, 100), weight=1.0, flip_pair=flip_name, sigma=0.040
        ))
    # Hand finger edges (wrist: 91, 5 fingers of 4 joints)
    for f in range(5):
        base = 91 + 1 + f * 4
        edges.append((91, base))
        edges.extend([(base, base + 1), (base + 1, base + 2), (base + 2, base + 3)])

    # 112-132: Right Hand (21)
    for i in range(21):
        j_id = 112 + i
        name = f"right_hand_{i}"
        flip_name = f"left_hand_{i}"
        joints.append(JointSpec(
            id=j_id, name=name, group="right_hand",
            color=(255, 100, 100), weight=1.0, flip_pair=flip_name, sigma=0.040
        ))
    for f in range(5):
        base = 112 + 1 + f * 4
        edges.append((112, base))
        edges.extend([(base, base + 1), (base + 1, base + 2), (base + 2, base + 3)])

    return KeypointSpec("coco_wholebody_133", joints, edges)


# Pre-register default specs
KEYPOINT_SPECS.register_module(create_coco_17_spec(), name="coco_17")
KEYPOINT_SPECS.register_module(create_coco_wholebody_133_spec(), name="coco_wholebody_133")
