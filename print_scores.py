import torch
import numpy as np
from wholebody.inference.api import init_model, PosePredictor

model = init_model(config="configs/experiments/rtmw-l_384x288.yaml", checkpoint="weights/rtmw-l_384x288_fixed.pth")
predictor = PosePredictor(model=model, input_size=(384, 288))

import cv2
img = cv2.imread("persona.jpeg")
results = predictor.predict(img)

scores = results[0]["keypoint_scores"]
print("Min score:", scores.min())
print("Max score:", scores.max())
print("Mean score:", scores.mean())
print(scores[:5])
