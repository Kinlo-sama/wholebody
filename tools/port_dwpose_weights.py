import argparse
import torch

def parse_args():
    parser = argparse.ArgumentParser(description='Extract student weights from DWPose distillation checkpoint')
    parser.add_argument('in_path', help='Path to the downloaded DWPose checkpoint')
    parser.add_argument('out_path', help='Path to save the extracted student checkpoint')
    return parser.parse_args()

def main():
    args = parse_args()
    print(f"Loading checkpoint from {args.in_path}...")
    ckpt = torch.load(args.in_path, map_location='cpu')
    
    # We explicitly extract from the regular state_dict.
    # DWPose's original training script appears to have failed to update the student
    # within the ema_state_dict, leaving it randomly initialized. The regular state_dict
    # correctly stores the fully trained student weights without any wrapper prefixes.
    state = ckpt.get('state_dict', ckpt)
    
    torch.save(state, args.out_path)
    print(f"Extracted {len(state)} weights.")
    print(f"Successfully saved clean student checkpoint to {args.out_path}")

if __name__ == '__main__':
    main()
