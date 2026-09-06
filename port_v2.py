import torch
import sys

def port_weights(in_path, out_path):
    ckpt = torch.load(in_path, map_location='cpu')
    # Force use of regular state_dict!
    state = ckpt.get('state_dict', ckpt)
    
    new_state = {}
    for k, v in state.items():
        if k.startswith('module.student.'):
            new_key = k.replace('module.student.', '')
            new_state[new_key] = v
        elif k.startswith('module.teacher.'):
            continue
        else:
            new_state[k] = v
            
    torch.save(new_state, out_path)
    print(f"Extracted {len(new_state)} student weights from state_dict.")

if __name__ == "__main__":
    port_weights("weights/dw-mm_ucoco.pth", "weights/dw-mm_ucoco_v2.pth")
