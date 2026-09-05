import numpy as np
import cv2

center = np.array([100.0, 150.0])
scale = np.array([1.5, 2.0]) # w_padded = 300, h_padded = 400
w_padded, h_padded = scale * 200.0

w_in, h_in = 192, 256

# MMPose direct formula
def mmpose_inv(pt):
    return pt / np.array([w_in, h_in]) * np.array([w_padded, h_padded]) + center - 0.5 * np.array([w_padded, h_padded])

# My affine transform
src_w = w_padded
dst_w = w_in
dst_h = h_in
src_dir = np.array([0, -0.5 * src_w])
dst_dir = np.array([0, -0.5 * dst_w])

src = np.zeros((3, 2), dtype=np.float32)
src[0] = center
src[1] = center + src_dir
src[2] = src[0] + np.array([-src_dir[1], src_dir[0]])

dst = np.zeros((3, 2), dtype=np.float32)
dst[0] = [dst_w * 0.5, dst_h * 0.5]
dst[1] = np.array([dst_w * 0.5, dst_h * 0.5]) + dst_dir
dst[2] = dst[0] + np.array([-dst_dir[1], dst_dir[0]])

warp_mat_inv = cv2.getAffineTransform(np.float32(dst), np.float32(src))

def my_inv(pt):
    return np.dot(np.append(pt, 1.0), warp_mat_inv.T)

pt = np.array([50.0, 70.0])
print("MMPose:", mmpose_inv(pt))
print("Mine:", my_inv(pt))
print("Diff:", np.abs(mmpose_inv(pt) - my_inv(pt)))
