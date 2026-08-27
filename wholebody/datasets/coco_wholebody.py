import json
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np

from wholebody.core.registry import DATASETS
from wholebody.datasets.base import BasePoseDataset
from wholebody.utils.logger import get_logger

logger = get_logger("wholebody.datasets.coco_wholebody")

@DATASETS.register("COCOWholeBodyDataset")
class COCOWholeBodyDataset(BasePoseDataset):
    """Dataset para cargar anotaciones COCO-WholeBody (133 keypoints)."""

    def __init__(
        self,
        ann_file: str,
        img_prefix: str,
        pipeline: List[Dict[str, Any]],
        keypoint_spec: str = "coco_wholebody_133",
        test_mode: bool = False,
    ) -> None:
        self.ann_file = ann_file
        self.img_prefix = Path(img_prefix)
        super().__init__(
            pipeline=pipeline,
            keypoint_spec=keypoint_spec,
            test_mode=test_mode
        )

    def load_annotations(self) -> List[Dict[str, Any]]:
        logger.info(f"Cargando anotaciones COCO-WholeBody desde {self.ann_file}...")
        
        with open(self.ann_file, 'r') as f:
            coco_data = json.load(f)

        images = {img['id']: img for img in coco_data['images']}
        data_list = []
        
        for ann in coco_data['annotations']:
            # Ignorar escenas de multitudes
            if ann.get('iscrowd', 0):
                continue
                
            # Verificar si tiene los keypoints base
            if 'keypoints' not in ann:
                continue

            img_id = ann['image_id']
            img_info = images[img_id]
            img_path = str(self.img_prefix / img_info['file_name'])
            
            # --- CONCATENACIÓN DE 133 KEYPOINTS ---
            # 1. Body (17)
            body_kpts = np.array(ann['keypoints'], dtype=np.float32).reshape(-1, 3)
            
            # 2. Foot (6)
            if 'foot_kpts' in ann and ann['foot_kpts'] is not None:
                foot_kpts = np.array(ann['foot_kpts'], dtype=np.float32).reshape(-1, 3)
            else:
                foot_kpts = np.zeros((6, 3), dtype=np.float32)
                
            # 3. Face (68)
            if 'face_kpts' in ann and ann['face_kpts'] is not None:
                face_kpts = np.array(ann['face_kpts'], dtype=np.float32).reshape(-1, 3)
            else:
                face_kpts = np.zeros((68, 3), dtype=np.float32)
                
            # 4. Left Hand (21)
            if 'lefthand_kpts' in ann and ann['lefthand_kpts'] is not None:
                lh_kpts = np.array(ann['lefthand_kpts'], dtype=np.float32).reshape(-1, 3)
            else:
                lh_kpts = np.zeros((21, 3), dtype=np.float32)
                
            # 5. Right Hand (21)
            if 'righthand_kpts' in ann and ann['righthand_kpts'] is not None:
                rh_kpts = np.array(ann['righthand_kpts'], dtype=np.float32).reshape(-1, 3)
            else:
                rh_kpts = np.zeros((21, 3), dtype=np.float32)

            # Unir todos para formar el tensor de (133, 3)
            all_kpts = np.concatenate([body_kpts, foot_kpts, face_kpts, lh_kpts, rh_kpts], axis=0)
            
            # Separar coordenadas (x,y) y visibilidad (v)
            kpts = all_kpts[:, :2].copy()
            vis = all_kpts[:, 2].copy()

            # Bounding Box (x, y, w, h) -> (x1, y1, x2, y2)
            bbox = np.array(ann['bbox'], dtype=np.float32)
            bbox_xyxy = np.array([bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]], dtype=np.float32)

            # Centro y escala para TopDownAffine (MMPose standard: scale by 200px)
            center = np.array([bbox[0] + bbox[2]/2.0, bbox[1] + bbox[3]/2.0], dtype=np.float32)
            scale = np.array([bbox[2]/200.0, bbox[3]/200.0], dtype=np.float32)
            
            # En COCO, a veces se expande la caja un 25% para que la persona no toque los bordes
            scale = scale * 1.25

            data_list.append({
                "img_path": img_path,
                "img_id": img_id,
                "ann_id": ann['id'],
                "img_shape": (img_info['height'], img_info['width'], 3),
                "raw_keypoints": kpts.copy(),
                "keypoints": kpts,
                "keypoints_visible": vis,
                "bboxes": bbox_xyxy.reshape(1, 4),
                "center": center,
                "scale": scale,
                "rotation": 0.0,
            })
            
        logger.info(f"Se cargaron {len(data_list)} personas válidas.")
        return data_list
