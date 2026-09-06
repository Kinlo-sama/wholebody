import copy
from typing import Any, Dict, List, Optional
import torch
from torch.utils.data import Dataset, ConcatDataset

from wholebody.core.registry import DATASETS
from wholebody.datasets.pipelines import Compose

@DATASETS.register("CombinedDataset")
class CombinedDataset(Dataset):
    """Dataset wrapper that concatenates multiple datasets and applies a single pipeline.
    
    This is commonly used in DWPose to mix COCO and UBody datasets during training.
    """
    def __init__(
        self,
        datasets: List[Dict[str, Any]],
        pipeline: List[Dict[str, Any]],
        metainfo: Optional[Dict[str, Any]] = None,
        test_mode: bool = False,
    ):
        super().__init__()
        self.test_mode = test_mode
        self.pipeline = Compose(pipeline)
        
        # Build individual datasets without their own pipelines
        self.datasets = []
        for cfg in datasets:
            cfg = copy.deepcopy(cfg)
            cfg['pipeline'] = []  # Bypass individual pipelines
            cfg['test_mode'] = test_mode
            self.datasets.append(DATASETS.build(cfg))
            
        self.concat_dataset = ConcatDataset(self.datasets)
        
        # Expose common properties needed by dataloader / metrics
        self.metainfo = metainfo or getattr(self.datasets[0], 'metainfo', {})
        self.keypoint_spec = getattr(self.datasets[0], 'keypoint_spec', None)
        
    def __len__(self) -> int:
        return len(self.concat_dataset)
        
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        data_info = self.concat_dataset[idx]
        results = self.pipeline(data_info)
        
        if results is None:
            # Fallback to next sample if pipeline drops this one (e.g., RandomHalfBody fail)
            return self.__getitem__((idx + 1) % len(self))
            
        return results
