import torch
import numpy as np
import cv2
import sys
import os

# Create dummy input data
B, K, Wx, Wy = 1, 133, 192*2, 256*2
simcc_x = torch.randn(B, K, Wx)
simcc_y = torch.randn(B, K, Wy)

# --- MMPOSE DECODER SIMULATION ---
def mmpose_decode(simcc_x, simcc_y):
    x_locs = simcc_x.argmax(dim=2).float()
    y_locs = simcc_y.argmax(dim=2).float()
    
    x_scores = simcc_x.amax(dim=2)
    y_scores = simcc_y.amax(dim=2)
    
    # get_simcc_maximum
    mask = x_scores > y_scores
    vals = x_scores.clone()
    vals[mask] = y_scores[mask]
    
    keypoints = torch.stack([x_locs, y_locs], dim=-1)
    
    # decode
    keypoints /= 2.0
    return keypoints.numpy(), vals.numpy()

def mmpose_inverse_warp(kpts, center, scale):
    w_padded, h_padded = scale * 200.0
    w_in, h_in = 192, 256
    # kpts: [B, K, 2]
    kpts = kpts / np.array([w_in, h_in]) * np.array([w_padded, h_padded]) + center - 0.5 * np.array([w_padded, h_padded])
    return kpts

m_kpts, m_scores = mmpose_decode(simcc_x, simcc_y)
center = np.array([200.0, 300.0])
scale = np.array([1.5, 2.0])
m_kpts_orig = mmpose_inverse_warp(m_kpts, center, scale)

# --- OUR DECODER SIMULATION ---
sys.path.insert(0, os.path.abspath('.'))
from wholebody.codecs.simcc_codec import SimCCCodec
from wholebody.datasets.transforms.affine import get_affine_transform

codec = SimCCCodec(input_size=(256, 192), simcc_split_ratio=2.0)
meta = [{
    "center": center,
    "scale": scale,
    "warp_mat_inv": get_affine_transform(center, scale, 0.0, (256, 192), inv=True)
}]

our_kpts_orig, our_scores = codec.decode((simcc_x, simcc_y), metainfo=meta)

# --- COMPARE ---
print("Scores diff:", np.abs(m_scores - our_scores).max())
print("Coords diff:", np.abs(m_kpts_orig - our_kpts_orig).max())
