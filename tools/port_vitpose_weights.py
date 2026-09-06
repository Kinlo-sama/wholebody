import argparse
import torch

def parse_args():
    parser = argparse.ArgumentParser(description='Convert ViTPose weights to wholebody format')
    parser.add_argument('in_path', help='Path to the downloaded ViTPose checkpoint')
    parser.add_argument('out_path', help='Path to save the converted checkpoint')
    return parser.parse_args()

def main():
    args = parse_args()
    print(f"Loading checkpoint from {args.in_path}...")
    ckpt = torch.load(args.in_path, map_location='cpu', weights_only=False)
    
    state = ckpt.get('state_dict', ckpt)
    new_state = {}
    
    for k, v in state.items():
        # Rename keypoint_head to head
        if k.startswith('keypoint_head.'):
            new_k = k.replace('keypoint_head.', 'head.')
        else:
            new_k = k
        new_state[new_k] = v
        
    torch.save(new_state, args.out_path)
    print(f"Extracted and renamed {len(new_state)} weights.")
    print(f"Successfully saved cleanly to {args.out_path}")

if __name__ == '__main__':
    main()
