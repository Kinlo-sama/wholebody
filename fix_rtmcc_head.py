import re

with open('wholebody/models/heads/rtmcc_head.py', 'r') as f:
    content = f.read()

# Replace RTMWHead with RTMCCHead
content = content.replace('HEADS.register("RTMWHead")', 'HEADS.register("RTMCCHead")')
content = content.replace('class RTMWHead(nn.Module):', 'class RTMCCHead(BaseHead):')
content = content.replace('super().__init__()', 'super().__init__()\n        from wholebody.models.losses.kl_discret_loss import KLDiscretLoss\n        self.loss_module = KLDiscretLoss(use_target_weight=True, beta=10.0, label_softmax=True)')

# We need to remove final_layer2, conv_dec, mlp2, ps
content = re.sub(r'        self.conv_dec =.*?\n', '', content)
content = re.sub(r'        self.ps =.*?\n', '', content)
content = re.sub(r'        self.final_layer2 =.*?\n', '', content)
content = re.sub(r'        self.mlp2 =.*?\n\s+ScaleNorm.*?\n\s+nn.Linear.*?\n', '', content)

# Forward pass change
forward_func = """    def forward(self, feats):
        if isinstance(feats, (list, tuple)):
            feats = feats[-1]
            
        feats = self.final_layer(feats)
        feats = torch.flatten(feats, 2)
        feats = self.mlp(feats)
        feats = self.gau(feats)
        
        pred_x = self.cls_x(feats)
        pred_y = self.cls_y(feats)
        
        return pred_x, pred_y"""

content = re.sub(r'    def forward\(self, feats: Tuple\[torch.Tensor\]\) -> Tuple\[torch.Tensor, torch.Tensor\]:.*?return pred_x, pred_y', forward_func, content, flags=re.DOTALL)

with open('wholebody/models/heads/rtmcc_head.py', 'w') as f:
    f.write(content)
