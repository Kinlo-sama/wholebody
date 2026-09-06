import re

with open('wholebody/models/heads/rtmcc_head.py', 'r') as f:
    content = f.read()

# Fix the garbage lines inside __init__
pattern = r"        flatten_dims = self.in_featuremap_size\[0\] \* self.in_featuremap_size\[1\]\n        ps = 2\n.*?        self.final_layer = ConvModule\("
replacement = "        flatten_dims = self.in_featuremap_size[0] * self.in_featuremap_size[1]\n        self.final_layer = ConvModule("

content = re.sub(pattern, replacement, content, flags=re.DOTALL)

pattern2 = r"            padding=final_layer_kernel_size // 2,\n            act=nn.ReLU\(inplace=True\)\n        \)\n        \n            in_channels // ps \+ in_channels // 4,\n            out_channels,\n            kernel_size=final_layer_kernel_size,\n            stride=1,\n            padding=final_layer_kernel_size // 2,\n            act=nn.ReLU\(inplace=True\)\n        \)"

replacement2 = "            padding=final_layer_kernel_size // 2,\n            act=nn.ReLU(inplace=True)\n        )"
content = re.sub(pattern2, replacement2, content, flags=re.DOTALL)

pattern3 = r"        self.mlp = nn.Sequential\(\n            ScaleNorm\(flatten_dims\),\n            nn.Linear\(flatten_dims, gau_cfg\['hidden_dims'\] // 2, bias=False\)\n        \)\n\n        \)"
replacement3 = "        self.mlp = nn.Sequential(\n            ScaleNorm(flatten_dims),\n            nn.Linear(flatten_dims, gau_cfg['hidden_dims'], bias=False)\n        )"
content = re.sub(pattern3, replacement3, content, flags=re.DOTALL)

with open('wholebody/models/heads/rtmcc_head.py', 'w') as f:
    f.write(content)
