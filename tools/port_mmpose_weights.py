import torch
import argparse
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(description="Port MMPose weights to Wholebody framework")
    parser.add_argument("input", type=str, help='Path to the original MMPose .pth file')
    parser.add_argument("output", type=str, help="Destination path for the port .pth file")
    return parser.parse_args()
    

def port_weights():
    args = parse_args()
    # Load the file into the RAM (CPU)
    # MMPose checkpoints save full config dicts and numpy arrays in 'meta', so we need weights_only=False
    mmpose_ckpt = torch.load(args.input, map_location='cpu', weights_only=False)

    # MMPOSE sometimes saves extra data (like the optimizer). We only want the "state_dict"
    state_dict = mmpose_ckpt.get('state_dict', mmpose_ckpt)
    
    # Translate the names
    our_state = {}
    for old_key, tensor in state_dict.items():
        new_key = old_key
        
        # RULE: chance "keypoint_head" to "head"
        if old_key.startswith('keypoint_head.'):
            new_key = old_key.replace('keypoint_head.', 'head.')
            
        our_state[new_key] = tensor

    # 5. Save the new file ready for your framework
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(our_state, out_path)
    print(f"Model successfully ported and saved to {out_path}!")

if __name__ == "__main__":
    port_weights()