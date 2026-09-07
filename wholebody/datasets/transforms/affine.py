import random
from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np

from wholebody.core.registry import TRANSFORMS
from wholebody.datasets.transforms.base import BaseTransform


def get_affine_transform(
    center: np.ndarray,
    scale: np.ndarray,
    rot: float,
    output_size: Tuple[int, int],  # (H, W)
    shift: Tuple[float, float] = (0.0, 0.0),
    inv: bool = False,
) -> np.ndarray:
    """Compute 2x3 affine transformation matrix mapping from bounding box to crop."""
    h_out, w_out = output_size
    scale_tmp = scale * 200.0
    src_w = scale_tmp[0]
    dst_w = float(w_out)
    dst_h = float(h_out)

    rot_rad = np.pi * rot / 180.0
    src_dir = np.array([0, -0.5 * src_w], dtype=np.float32)
    dst_dir = np.array([0, -0.5 * dst_w], dtype=np.float32)

    sn, cs = np.sin(rot_rad), np.cos(rot_rad)
    src_dir_rot = np.array([src_dir[0] * cs - src_dir[1] * sn, src_dir[0] * sn + src_dir[1] * cs], dtype=np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)

    src[0, :] = center + np.array(shift, dtype=np.float32) * scale_tmp
    src[1, :] = center + src_dir_rot + np.array(shift, dtype=np.float32) * scale_tmp
    # Third point to maintain aspect ratio and orientation
    src_dir_perp = np.array([-src_dir_rot[1], src_dir_rot[0]], dtype=np.float32)
    src[2, :] = src[0, :] + src_dir_perp

    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5]) + dst_dir
    dst_dir_perp = np.array([-dst_dir[1], dst_dir[0]], dtype=np.float32)
    dst[2, :] = dst[0, :] + dst_dir_perp

    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    return trans



def get_warp_matrix(theta: float, size_input: np.ndarray, size_dst: np.ndarray, size_target: np.ndarray) -> np.ndarray:
    theta = np.deg2rad(theta)
    matrix = np.zeros((2, 3), dtype=np.float32)
    scale_x = size_dst[0] / size_target[0]
    scale_y = size_dst[1] / size_target[1]
    matrix[0, 0] = np.cos(theta) * scale_x
    matrix[0, 1] = -np.sin(theta) * scale_x
    matrix[0, 2] = scale_x * (-0.5 * size_input[0] * np.cos(theta) +
                              0.5 * size_input[1] * np.sin(theta) +
                              0.5 * size_target[0])
    matrix[1, 0] = np.sin(theta) * scale_y
    matrix[1, 1] = np.cos(theta) * scale_y
    matrix[1, 2] = scale_y * (-0.5 * size_input[0] * np.sin(theta) -
                              0.5 * size_input[1] * np.cos(theta) +
                              0.5 * size_target[1])
    return matrix

@TRANSFORMS.register("TopDownAffine")

class TopDownAffine(BaseTransform):
    """Crop and warp image and keypoints centered on person bounding box into model input shape."""

    def __init__(self, input_size: Tuple[int, int] = (256, 192), use_udp: bool = False) -> None:
        self.input_size = tuple(input_size)
        self.use_udp = use_udp  # (H, W)

    def transform(self, results: Dict[str, Any]) -> Dict[str, Any]:
        img = results["img"]
        center = results.get("center")
        scale = results.get("scale")

        if center is None or scale is None:
            # Default to full image bbox
            h, w = img.shape[:2]
            center = np.array([w / 2.0, h / 2.0], dtype=np.float32)
            scale = np.array([w / 200.0, h / 200.0], dtype=np.float32)
            results["center"] = center
            results["scale"] = scale

        rot = results.get("rotation", 0.0)
        h_in, w_in = self.input_size

                # Fix scale aspect ratio to prevent cropping tall/wide people!
        aspect_ratio = float(w_in) / float(h_in)
        if scale[0] > aspect_ratio * scale[1]:
            scale[1] = scale[0] / aspect_ratio
        elif scale[0] < aspect_ratio * scale[1]:
            scale[0] = scale[1] * aspect_ratio

        if self.use_udp:
            warp_mat = get_warp_matrix(rot, center * 2.0, np.array([w_in - 1.0, h_in - 1.0]), scale * 200.0)
            warp_mat_inv = get_warp_matrix(-rot, np.array([w_in - 1.0, h_in - 1.0]), center * 2.0, scale * 200.0)
            # Need actual inverse for proper mapping back, get_warp_matrix doesn't natively invert center/scale the same way.
            # Instead, let's just invert the 2x3 matrix using cv2.invertAffineTransform
            warp_mat_inv = cv2.invertAffineTransform(warp_mat)
        else:
            warp_mat = get_affine_transform(center, scale, rot, (h_in, w_in))
            warp_mat_inv = get_affine_transform(center, scale, rot, (h_in, w_in), inv=True)

        warped_img = cv2.warpAffine(img, warp_mat, (w_in, h_in), flags=cv2.INTER_LINEAR)
        results["img"] = warped_img
        results["input_size"] = (h_in, w_in)
        results["warp_mat"] = warp_mat
        results["warp_mat_inv"] = warp_mat_inv

        # Transform keypoints
        if "keypoints" in results:
            kpts = results["keypoints"].copy()
            num_kpts = kpts.shape[0]
            pts_homo = np.concatenate([kpts[:, :2], np.ones((num_kpts, 1), dtype=np.float32)], axis=1)
            warped_kpts = np.dot(pts_homo, warp_mat.T)[:, :2]
            results["keypoints"] = warped_kpts

        return results


@TRANSFORMS.register("RandomFlip")
class RandomFlip(BaseTransform):
    """Horizontally flip image and swap symmetric keypoints."""

    def __init__(self, prob: float = 0.5) -> None:
        self.prob = prob

    def transform(self, results: Dict[str, Any]) -> Dict[str, Any]:
        if random.random() > self.prob:
            return results

        img = results["img"]
        w = img.shape[1]
        results["img"] = cv2.flip(img, 1)

        # Update center if present
        if "center" in results:
            results["center"][0] = w - 1 - results["center"][0]

        # Flip keypoints
        if "keypoints" in results:
            kpts = results["keypoints"].copy()
            kpts[:, 0] = w - 1 - kpts[:, 0]

            # Reorder keypoint indices based on flip mapping
            flip_indices = results.get("flip_indices")
            if flip_indices is not None:
                kpts = kpts[flip_indices]
                if "keypoints_visible" in results:
                    results["keypoints_visible"] = results["keypoints_visible"][flip_indices]

            results["keypoints"] = kpts

        results["flipped"] = True
        return results


@TRANSFORMS.register("RandomRotation")
class RandomRotation(BaseTransform):
    """Add random rotation angle to data sample."""

    def __init__(self, max_angle: float = 30.0, prob: float = 0.6) -> None:
        self.max_angle = max_angle
        self.prob = prob

    def transform(self, results: Dict[str, Any]) -> Dict[str, Any]:
        if random.random() <= self.prob:
            rot = random.uniform(-self.max_angle, self.max_angle)
            results["rotation"] = results.get("rotation", 0.0) + rot
        return results
