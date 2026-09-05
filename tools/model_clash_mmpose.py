import argparse
import torch

try:
    from mmengine.config import Config as MMConfig
    from mmpose.registry import MODELS as MM_MODELS
except ImportError:
    print("Error: Instala mmengine y mmpose en este entorno.")
    import sys
    sys.exit(1)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, help="Ruta al config de MMPose")
    parser.add_argument('--checkpoint', required=True, help="Ruta al pth original")
    parser.add_argument('--out', default='mmpose_outputs.pt', help="Donde guardar el output")
    args = parser.parse_args()

    print("Cargando MMPose...")
    cfg = MMConfig.fromfile(args.config)
    model = MM_MODELS.build(cfg.model)
    ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    
    # Extraer state_dict si es necesario
    state_dict = ckpt.get('state_dict', ckpt)
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    torch.manual_seed(42)
    dummy_input = torch.randn(1, 3, 256, 192)

    print("Ejecutando Forward Pass...")
    with torch.no_grad():
        feats = model.extract_feat(dummy_input)
        pred_x, pred_y = model.head.forward(feats)

    torch.save({
        'feats': feats,
        'pred_x': pred_x,
        'pred_y': pred_y
    }, args.out)
    
    print(f"¡Listo! Outputs guardados en {args.out}")

if __name__ == '__main__':
    main()
