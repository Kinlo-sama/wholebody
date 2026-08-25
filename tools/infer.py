import argparse
from pathlib import Path
import cv2
import numpy as np

from wholebody.core.config import Config
from wholebody.core.registry import KEYPOINT_SPECS
from wholebody.inference.api import PosePredictor, init_model
from wholebody.visualization.visualizer import SkeletonVisualizer
from wholebody.utils.logger import get_logger

logger = get_logger("wholebody.tools.infer")


def parse_args():
    parser = argparse.ArgumentParser(description="Run Pose Estimation Inference")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint (.pth)")
    parser.add_argument("--input", type=str, default=None, help="Path to input image (or None to generate demo image)")
    parser.add_argument("--output", type=str, default="./work_dirs/pred.jpg", help="Path to save output visual result")
    parser.add_argument("--device", type=str, default="auto", help="Device ('auto', 'mps', 'cuda', 'cpu')")
    parser.add_argument("--keypoint-spec", type=str, default="coco_wholebody_133", help="KeypointSpec preset name")
    return parser.parse_args()


def main():
    args = parse_args()

    # Initialize model
    model = init_model(config=args.config, checkpoint=args.checkpoint, device=args.device)
    predictor = PosePredictor(model)

    # Initialize visualizer
    visualizer = SkeletonVisualizer(keypoint_spec=args.keypoint_spec)

    # Load or generate demo input image
    if args.input is not None and Path(args.input).is_file():
        img_bgr = cv2.imread(args.input)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    else:
        logger.info("No input image specified. Generating demo canvas with synthetic person...")
        from wholebody.datasets.synthetic import SyntheticWholeBodyDataset
        ds = SyntheticWholeBodyDataset(num_samples=1, keypoint_spec=args.keypoint_spec)
        img_rgb = ds.data_list[0]["img"]

    # Predict
    result_sample = predictor.predict(img_rgb)

    # Visualize
    drawn_rgb = visualizer.draw_sample(img_rgb, result_sample)
    drawn_bgr = cv2.cvtColor(drawn_rgb, cv2.COLOR_RGB2BGR)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), drawn_bgr)
    logger.info(f"Inference result saved to: {out_path}")


if __name__ == "__main__":
    main()
