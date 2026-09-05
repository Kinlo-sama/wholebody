import argparse
import torch
import sys
import os

# Asegurar que encuentre la carpeta wholebody
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from wholebody.core.config import Config as WBConfig
from wholebody.inference.api import init_model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, help="Ruta a nuestro config yaml")
    parser.add_argument('--checkpoint', required=True, help="Ruta a nuestro pth porteado")
    parser.add_argument('--mmpose-out', default='mmpose_outputs.pt', help="El archivo generado por MMPose")
    args = parser.parse_args()

    print("Cargando modelo WholeBody...")
    cfg = WBConfig.from_file(args.config)
    model = init_model(config=cfg, checkpoint=args.checkpoint, device='cpu')
    model.eval()

    torch.manual_seed(42)
    dummy_input = torch.randn(1, 3, 256, 192)

    print("Ejecutando Forward Pass...")
    with torch.no_grad():
        wb_feats = model.extract_feat(dummy_input)
        wb_pred_x, wb_pred_y = model.head.forward(wb_feats)

    print(f"Cargando tensores originales de {args.mmpose_out}...")
    mm_data = torch.load(args.mmpose_out, map_location='cpu', weights_only=False)
    mm_feats = mm_data['feats']
    mm_pred_x = mm_data['pred_x']
    mm_pred_y = mm_data['pred_y']

    print("\n--- Comparación Matemática ---")
    for i in range(len(mm_feats)):
        diff = torch.abs(mm_feats[i] - wb_feats[i]).max().item()
        print(f"Feat[{i}] (Backbone+Neck) Max Diff: {diff:.6f}")

    diff_x = torch.abs(mm_pred_x - wb_pred_x).max().item()
    diff_y = torch.abs(mm_pred_y - wb_pred_y).max().item()
    print(f"Pred X (Head) Max Diff: {diff_x:.6f}")
    print(f"Pred Y (Head) Max Diff: {diff_y:.6f}")

    if diff_x < 1e-4 and diff_y < 1e-4:
        print("\n¡Veredicto: MATEMÁTICAMENTE IDÉNTICOS!")
        print("El forward pass es perfecto. El error de 4.6 AP debe estar en el test pipeline (BBoxes, TTA, post-procesamiento de SimCC, o Evaluación Coco).")
    else:
        print("\n¡Veredicto: DIVERGENCIA DETECTADA!")
        print("El error de AP está causado por diferencias en la arquitectura (una capa actuando diferente).")

if __name__ == '__main__':
    main()
