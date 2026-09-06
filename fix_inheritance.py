with open('wholebody/models/heads/rtmcc_head.py', 'r') as f:
    content = f.read()

content = content.replace('from wholebody.models.heads.base import BaseHead\n', '')
content = content.replace('class RTMCCHead(BaseHead):', 'class RTMCCHead(nn.Module):')

with open('wholebody/models/heads/rtmcc_head.py', 'w') as f:
    f.write(content)
