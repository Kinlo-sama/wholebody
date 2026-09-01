with open('wholebody/evaluation/coco_metric.py', 'r') as f:
    content = f.read()

# I need to insert "    def reset(self) -> None:\n        self.results = []\n" after __init__
init_end_idx = content.find("        self.results: List[Dict] = []")
if init_end_idx != -1:
    content = content.replace("        self.results: List[Dict] = []\n", "        self.results: List[Dict] = []\n\n    def reset(self) -> None:\n        self.results = []\n")
    
with open('wholebody/evaluation/coco_metric.py', 'w') as f:
    f.write(content)
