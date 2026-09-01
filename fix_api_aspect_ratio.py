with open('wholebody/inference/api.py', 'r') as f:
    content = f.read()

fix_code = """
        # Fix aspect ratio
        aspect_ratio = self.input_size[0] / float(self.input_size[1]) # W/H in MMPose is input_size=(288, 384)? Wait.
        # Let's check self.input_size. Typically input_size is (W, H) or (H, W).
        # In PosePredictor: self.input_size is (384, 288) or (288, 384).
        # In get_affine_transform: h_out, w_out = output_size. So input_size is (H, W).
        # Aspect ratio is W / H.
        aspect_ratio = self.input_size[1] / float(self.input_size[0])
        w, h = scale[0], scale[1]
        if w > h * aspect_ratio:
            scale[1] = w / aspect_ratio
        else:
            scale[0] = h * aspect_ratio
"""

if "# Fix aspect ratio" not in content:
    content = content.replace("scale = np.array([box_w / 200.0, box_h / 200.0], dtype=np.float32)", 
                              "scale = np.array([box_w / 200.0, box_h / 200.0], dtype=np.float32)\n" + fix_code)
    with open('wholebody/inference/api.py', 'w') as f:
        f.write(content)
