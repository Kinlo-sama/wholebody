from pathlib import Path
from typing import Any, Callable, Optional, Union
import cv2
import numpy as np

from wholebody.visualization.visualizer import SkeletonVisualizer


class VideoVisualizer:
    """Processes and renders keypoints on video frames."""

    def __init__(self, visualizer: SkeletonVisualizer) -> None:
        self.visualizer = visualizer

    def render_video(
        self,
        video_path: Union[str, Path],
        output_path: Union[str, Path],
        frame_process_fn: Callable[[np.ndarray], Any],
        max_frames: Optional[int] = None,
    ) -> None:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise IOError(f"Cannot open video file: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            data_sample = frame_process_fn(frame)
            drawn_frame = self.visualizer.draw_sample(frame, data_sample)
            out.write(drawn_frame)

            frame_count += 1
            if max_frames and frame_count >= max_frames:
                break

        cap.release()
        out.release()
