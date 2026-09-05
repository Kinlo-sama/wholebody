import sys
sys.path.append('mmpose')
from mmpose.datasets.datasets.wholebody.coco_wholebody_dataset import CocoWholeBodyDataset

# The dataset needs some args
ds = CocoWholeBodyDataset(
    data_root='data/coco/',
    data_mode='topdown',
    ann_file='annotations/coco_wholebody_val_v1.0.json',
    data_prefix=dict(img='val2017/'),
    pipeline=[]
)
print("MMPose flip_indices:", ds.metainfo['flip_indices'])
