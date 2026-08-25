# WholeBody: A Modular Research Framework for Whole-Body Human Pose Estimation

**WholeBody** is a modular, research-oriented PyTorch framework designed for **Whole-Body Human Pose Estimation** (body, face, hands, and feet). Built with first-class support for **Apple Silicon (MPS)**, **NVIDIA CUDA**, and **CPU**, it decouples datasets, model architectures, keypoint topologies, and training engines for effortless academic exploration.

---

## Key Features

1. **First-Class Apple Silicon (MPS) Support**: Seamless acceleration on M-series chips with automatic device resolution (`CUDA -> MPS -> CPU`), AMP support, and cross-platform checkpoint portability.
2. **Universal Keypoint Spec**: Define and switch between 17, 133, 136, or custom skeletons without modifying backbones or training loops.
3. **Decoupled Codecs**: Independent encoding (heatmaps, SimCC, regression) and sub-pixel decoding.
4. **Declarative PyYAML Configuration**: Transparent, readable YAML with recursive `_base_` inheritance and CLI overrides.
5. **ModelGraph IR**: Intermediate DAG representation enabling tensor shape verification and future Visual GUI Builder integration.
6. **Robust Fine-Tuning**: Freeze backbones, replace heads, and adapt checkpoints with mismatched keypoint counts automatically.

---

## Quickstart

### 1. Installation
```bash
cd wholebody
pip install -e .
```

### 2. Run Tests
```bash
# Run all unit tests
python3 -m unittest discover -s tests -p "test_*.py" -v

# Run dedicated Apple Silicon MPS test
python3 tests/runtime/test_mps.py
```

### 3. Train a Model (Apple Silicon / MPS Demo)
```bash
python3 tools/train.py configs/experiments/demo_apple_silicon.yaml --epochs 3
```

### 4. Run Inference
```bash
python3 tools/infer.py \
    --config configs/experiments/demo_apple_silicon.yaml \
    --checkpoint work_dirs/demo_experiment/checkpoints/latest.pth \
    --output work_dirs/demo_experiment/pred.jpg
```

### 5. Export Architecture Graph (for GUI Visual Builders)
```bash
python3 tools/export_graph.py \
    --config configs/experiments/demo_apple_silicon.yaml \
    --output work_dirs/demo_experiment/model_graph.json
```
