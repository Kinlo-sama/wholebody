import numpy as np
import cv2

def get_affine_transform(
    center: np.ndarray,
    scale: np.ndarray,
    rot: float,
    output_size: tuple,  # (H, W)
    shift: tuple = (0.0, 0.0),
    inv: bool = False,
) -> np.ndarray:
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

c = np.array([100, 100])
s = np.array([1.0, 1.0])
out = (384, 288)

mat = get_affine_transform(c, s, 0, out)
print("My Mat:")
print(mat)

# Simple equivalent:
# We map center [100, 100] to [144, 192]
# We map box width 200 (x: 0 to 200) to 288 (scale = 288/200 = 1.44)
print("\nExpected:")
print(f"Scale X: {288/200}")
print(f"Scale Y: {288/200} (because aspect ratio forces Y to scale same as X?)")

