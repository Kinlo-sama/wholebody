import cv2
import torch
import numpy as np
from wholebody.core.config import Config
from wholebody.core.registry import MODELS
from wholebody.datasets.transforms.loading import LoadImageFromFile
from wholebody.datasets.transforms.affine import TopDownAffine
from wholebody.datasets.transforms.formatting import Normalize

# Load model
cfg = Config.from_file("configs/experiments/dwpose-m_256x192.yaml")
model = MODELS.build(cfg.model)
ckpt = torch.load("weights/dw-mm_ucoco_ported.pth", map_location='cpu')
model.load_state_dict(ckpt, strict=False)
model.eval()

# Fake annotation for an image
img_path = "data/coco/val2017/000000000139.jpg"
img = cv2.imread(img_path)
h, w = img.shape[:2]
results = {
    "img_path": img_path,
    "center": np.array([w/2.0, h/2.0], dtype=np.float32),
    "scale": np.array([w/200.0, h/200.0], dtype=np.float32) * 1.25,
}

# Pipeline
loader = LoadImageFromFile(to_rgb=True)
affine = TopDownAffine(input_size=(256, 192))
norm = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

results = loader.transform(results)
results = affine.transform(results)
results = norm.transform(results)

# Tensor
x = torch.from_numpy(results["img"]).permute(2, 0, 1).unsqueeze(0).float()

with torch.no_grad():
    feats = model.extract_feat(x)
    pred_x, pred_y = model.head(feats)
    
    # decode
    pred_coords, scores = model.head.codec.decode((pred_x, pred_y), [results])

# Draw on original image
kpts = pred_coords[0]
for kpt in kpts:
    x, y = int(kpt[0]), int(kpt[1])
    cv2.circle(img, (x, y), 3, (0, 0, 255), -1)

cv2.imwrite("test_vis.jpg", img)
print("Saved to test_vis.jpg")
