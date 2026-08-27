import torch

def port_weights():
    # 1. Cargar el archivo original que descargaste
    mmpose_ckpt = torch.load('weights/res50_coco_wholebody_256x192-9e37ed88_20201004.pth', map_location='cpu')
    
    # Extraemos el diccionario de pesos
    # A veces MMPose guarda el dict dentro de una llave 'state_dict'
    state_dict = mmpose_ckpt.get('state_dict', mmpose_ckpt)
    
    our_state = {}
    
    for old_key, tensor in state_dict.items():
        new_key = old_key
        
        # MMPose usa 'keypoint_head', nuestro framework usa 'head'
        if old_key.startswith('keypoint_head.'):
            new_key = old_key.replace('keypoint_head.', 'head.')
            
        our_state[new_key] = tensor

    # 2. Guardar el nuevo archivo compatible
    torch.save(our_state, 'nuestro_resnet50_coco.pth')
    print("¡Modelo porteado exitosamente!")

if __name__ == "__main__":
    port_weights()