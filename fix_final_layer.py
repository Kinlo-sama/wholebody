import re

with open('wholebody/models/heads/rtmcc_head.py', 'r') as f:
    content = f.read()

replacement = """        self.final_layer = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=final_layer_kernel_size,
            stride=1,
            padding=final_layer_kernel_size // 2
        )"""

pattern = r"        self\.final_layer = ConvModule\(\n.*?\n.*?\n.*?\n.*?\n.*?\n.*?\n        \)"

new_content = re.sub(pattern, replacement, content, flags=re.DOTALL)

with open('wholebody/models/heads/rtmcc_head.py', 'w') as f:
    f.write(new_content)
