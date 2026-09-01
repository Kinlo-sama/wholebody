import cv2
import numpy as np
import torch
import sys

from wholebody.inference.api import init_model, PosePredictor

def main():
    print("Loading model...")
    config_file = "configs/experiments/rtmw-l_384x288.yaml"
    checkpoint = "weights/rtmw-l_384x288_fixed.pth"
    image_path = "persona.jpeg"
    output_path = "persona_result.jpg"
    
    # Initialize the model and predictor
    model = init_model(config=config_file, checkpoint=checkpoint)
    predictor = PosePredictor(model=model, input_size=(384, 288))
    
    print(f"Running inference on {image_path}...")
    # Read the image
    orig_img = cv2.imread(image_path)
    if orig_img is None:
        print(f"Error: Could not load image from {image_path}")
        sys.exit(1)
        
    # Get predictions
    pred_sample = predictor.predict(image_path)
    
    # Extract keypoints and scores
    pred_instances = pred_sample.pred_instances
    keypoints = pred_instances.keypoints.detach().cpu().numpy()
    scores = pred_instances.keypoint_scores.detach().cpu().numpy()
    
    print(f"Predicted shape: {keypoints.shape}")
    print("First 5 Keypoints:")
    for i in range(5):
        print(f"  {i}: x={keypoints[i][0]:.2f}, y={keypoints[i][1]:.2f}, score={scores[i]:.2f}")
    
    # Draw keypoints on the image
    img_draw = orig_img.copy()
    
    # Optional: Connect body joints (simple skeleton just to see the structure)
    # COCO 17 body keypoints are the first 17
    # For now, just scatter all keypoints to see their distribution
    for i in range(len(keypoints)):
        x, y = int(keypoints[i, 0]), int(keypoints[i, 1])
        score = scores[i]
        
        # Color coding: 
        # 0-16: Body (Green)
        # 17-22: Feet (Yellow)
        # 23-90: Face (Blue)
        # 91-111: Left Hand (Red)
        # 112-132: Right Hand (Magenta)
        if i < 17:
            color = (0, 255, 0)
            r = 3
        elif i < 23:
            color = (0, 255, 255)
            r = 2
        elif i < 91:
            color = (255, 0, 0)
            r = 1
        elif i < 112:
            color = (0, 0, 255)
            r = 2
        else:
            color = (255, 0, 255)
            r = 2
            
        cv2.circle(img_draw, (x, y), r, color, -1)
        
    cv2.imwrite(output_path, img_draw)
    print(f"Result saved to {output_path}")

if __name__ == "__main__":
    main()
