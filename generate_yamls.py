import os

TEMPLATE = """_base_:
  - ../runtime/apple_silicon.yaml

experiment_name: {name}

training:
  epochs: 270
  val_interval: 10
  optimizer:
    type: AdamW
    lr: 0.004
    weight_decay: 0.05

model:
  type: TopDownPoseEstimator
  pretrained: "{ckpt}"
  test_cfg:
    flip_test: true
    shift_heatmap: false
  backbone:
    type: CSPNeXt
    arch: P5
    expand_ratio: 0.5
    deepen_factor: {deepen}
    widen_factor: {widen}
    channel_attention: true
    out_indices: [2, 3, 4]
  neck:
    type: CSPNeXtPAFPN
    in_channels: {neck_channels}
    out_channels: null
    out_indices: [1, 2]
    num_csp_blocks: 2
    expand_ratio: 0.5
  head:
    type: RTMWHead
    in_channels: {head_channels}
    out_channels: 133
    input_size: [{H}, {W}]
    in_featuremap_size: [{H32}, {W32}]
    simcc_split_ratio: 2.0
    final_layer_kernel_size: 7
    gau_cfg:
      hidden_dims: 256
      s: 128
      expansion_factor: 2
      dropout_rate: 0.
      drop_path: 0.
      act_fn: SiLU
      use_rel_bias: false
      pos_enc: false
    codec:
      type: SimCCLabel
      input_size: [{H}, {W}]
      sigma: {sigma}
      simcc_split_ratio: 2.0
      normalize: false

data:
  val:
    dataset:
      type: COCOWholeBodyDataset
      ann_file: data/coco/annotations/coco_wholebody_val_v1.0.json
      img_prefix: data/coco/val2017/
      keypoint_spec: coco_wholebody_133
      test_mode: true
      bbox_file: data/coco/person_detection_results/COCO_val2017_detections_AP_H_56_person.json
      pipeline:
        - type: LoadImageFromFile
        - type: TopDownAffine
          input_size: [{H}, {W}]
        - type: Normalize
          mean: [0.485, 0.456, 0.406]
          std: [0.229, 0.224, 0.225]
        - type: PackPoseInputs
    dataloader:
      batch_size: 32
      num_workers: 0
      shuffle: false

evaluation:
  metrics:
    - type: CocoWholeBodyMetric
      ann_file: data/coco/annotations/coco_wholebody_val_v1.0.json
"""

models = [
    {"name": "rtmw-m_256x192", "ckpt": "weights/rtmw-m_ported.pth", "deepen": 0.67, "widen": 0.75, "neck": "[192, 384, 768]", "head": 768, "H": 256, "W": 192, "sigma": 5.0},
    {"name": "rtmw-l_256x192", "ckpt": "weights/rtmw-l_ported.pth", "deepen": 1.0, "widen": 1.0, "neck": "[256, 512, 1024]", "head": 1024, "H": 256, "W": 192, "sigma": 5.0},
    {"name": "rtmw-l_384x288", "ckpt": "weights/rtmw-l_384x288_ported.pth", "deepen": 1.0, "widen": 1.0, "neck": "[256, 512, 1024]", "head": 1024, "H": 384, "W": 288, "sigma": 6.0},
    {"name": "rtmw-x_256x192", "ckpt": "weights/rtmw-x_ported.pth", "deepen": 1.33, "widen": 1.25, "neck": "[320, 640, 1280]", "head": 1280, "H": 256, "W": 192, "sigma": 5.0},
    {"name": "rtmw-x_384x288", "ckpt": "weights/rtmw-x_384x288_ported.pth", "deepen": 1.33, "widen": 1.25, "neck": "[320, 640, 1280]", "head": 1280, "H": 384, "W": 288, "sigma": 6.0},
]

for m in models:
    content = TEMPLATE.format(
        name=m["name"],
        ckpt=m["ckpt"],
        deepen=m["deepen"],
        widen=m["widen"],
        neck_channels=m["neck"],
        head_channels=m["head"],
        H=m["H"],
        W=m["W"],
        H32=m["H"] // 32,
        W32=m["W"] // 32,
        sigma=m["sigma"]
    )
    with open(f"configs/experiments/{m['name']}.yaml", "w") as f:
        f.write(content)

print("Generated 5 configuration files.")
