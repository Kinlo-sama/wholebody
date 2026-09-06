import re

with open('test_dummy_distiller.py', 'r') as f:
    content = f.read()
content = content.replace(", use_dark=False", "")
with open('test_dummy_distiller.py', 'w') as f:
    f.write(content)

with open('configs/experiments/dwpose-m_256x192.yaml', 'r') as f:
    yaml_content = f.read()
yaml_content = yaml_content.replace("      use_dark: false\n", "")
with open('configs/experiments/dwpose-m_256x192.yaml', 'w') as f:
    f.write(yaml_content)
