import numpy as np
import cv2
import sys
import os

# --- MMPOSE PREPROCESSING ---
def mmpose_preprocess(img, bbox):
    # GetBBoxCenterScale
    w = bbox[2]
    h = bbox[3]
    center = np.array([bbox[0] + w/2, bbox[1] + h/2])
    scale = np.array([w * 1.25, h * 1.25])
    
    # TopdownAffine _fix_aspect_ratio
    aspect_ratio = 192.0 / 256.0
    if scale[0] > aspect_ratio * scale[1]:
        scale[1] = scale[0] / aspect_ratio
    elif scale[0] < aspect_ratio * scale[1]:
        scale[0] = scale[1] * aspect_ratio
        
    # get_warp_matrix
    src_w = scale[0]
    dst_w = 192
    dst_h = 256
    
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
    
    warp_mat = cv2.getAffineTransform(src, dst)
    
    img_warp = cv2.warpAffine(img, warp_mat, (192, 256), flags=cv2.INTER_LINEAR)
    return img_warp

# --- OUR PREPROCESSING ---
sys.path.insert(0, os.path.abspath('.'))
from wholebody.datasets.transforms.affine import TopDownAffine

def our_preprocess(img, bbox):
    # Fake dataset dict
    results = {
        "img": img,
        "center": np.array([bbox[0] + bbox[2]/2.0, bbox[1] + bbox[3]/2.0], dtype=np.float32),
        "scale": np.array([bbox[2]/200.0, bbox[3]/200.0], dtype=np.float32) * 1.25
    }
    transform = TopDownAffine(input_size=(256, 192))
    res = transform.transform(results)
    return res["img"]

# Generate dummy image
img = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
bbox = [100.0, 50.0, 200.0, 300.0]

m_img = mmpose_preprocess(img, bbox)
our_img = our_preprocess(img, bbox)

print("Image Diff:", np.abs(m_img.astype(float) - our_img.astype(float)).max())
