import torch
import argparse
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(description="Port DWPose Distiller weights to Wholebody framework")
    parser.add_argument("input", type=str, help='Path to the original DWPose .pth file')
    parser.add_argument("output", type=str, help="Destination path for the port .pth file")
    return parser.parse_args()

def port_weights():
    args = parse_args()
    ckpt = torch.load(args.input, map_location='cpu', weights_only=False)

    # Prefer EMA weights as they almost always yield better AP
    if 'ema_state_dict' in ckpt:
        state_dict = ckpt['ema_state_dict']
        print("Using ema_state_dict for optimal AP.")
    else:
        state_dict = ckpt.get('state_dict', ckpt)
        print("Using standard state_dict.")
    
    our_state = {}
    for old_key, tensor in state_dict.items():
        new_key = old_key
        
        # If it's inside the distiller (EMA), extract the student
        if new_key.startswith('module.student.'):
            new_key = new_key.replace('module.student.', '')
        # Ignore the teacher
        elif new_key.startswith('module.teacher.'):
            continue
            
        # Legacy failsafe
        if new_key.startswith('keypoint_head.'):
            new_key = new_key.replace('keypoint_head.', 'head.')
            
        our_state[new_key] = tensor

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(our_state, out_path)
    print(f"DWPose weights successfully extracted and saved to {out_path}!")

if __name__ == "__main__":
    port_weights()
