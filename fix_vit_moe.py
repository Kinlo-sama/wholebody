with open("wholebody/models/backbones/vit_moe.py", "r") as f:
    text = f.read()

# Replace forward signature and logic
old_forward = """    def forward(self, x, dataset_source=None):
        x = self.forward_features(x, dataset_source)
        return x"""

new_forward = """    def forward(self, x, dataset_source=None):
        if dataset_source is None:
            # Default to WholeBody dataset_idx (5) if not provided
            dataset_source = torch.full((x.shape[0],), 5, dtype=torch.long, device=x.device)
        x = self.forward_features(x, dataset_source)
        return tuple([x])""" # wrap in tuple because RTMCCHead/HeatmapHead expects tuple from backbone

text = text.replace(old_forward, new_forward)

# Replace BACKBONES.register_module() with BACKBONES.register()
text = text.replace("BACKBONES.register_module()", 'BACKBONES.register("ViTMoE")')

with open("wholebody/models/backbones/vit_moe.py", "w") as f:
    f.write(text)
