import torch
import argparse
import os

def convert_rtmw_weights(input_path, output_path):
    print(f"Loading {input_path}...")
    ckpt = torch.load(input_path, map_location='cpu', weights_only=False)
    
    # Extract state dict (prefer EMA weights if available, as they are typically the best)
    if 'ema_state_dict' in ckpt:
        print("Found 'ema_state_dict'! Using EMA weights (best performance)...")
        state_dict = ckpt['ema_state_dict']
    else:
        print("No EMA weights found. Using standard 'state_dict'...")
        state_dict = ckpt.get('state_dict', ckpt)
    
    new_state_dict = {}
    
    print("Mapping keys...")
    for k, v in state_dict.items():
        new_k = k
        # MMPose uses 'data_preprocessor' which we don't need in our model
        if new_k.startswith('data_preprocessor.'):
            continue
            
        new_state_dict[new_k] = v

    print(f"Saving converted weights to {output_path}...")
    torch.save({"state_dict": new_state_dict}, output_path)
    print("Done!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True, help='Path to MMPose RTMW weights')
    parser.add_argument('--output', type=str, required=True, help='Output path')
    args = parser.parse_args()
    
    convert_rtmw_weights(args.input, args.output)
