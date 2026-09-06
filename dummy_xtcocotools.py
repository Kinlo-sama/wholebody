import sys
from unittest.mock import MagicMock
sys.modules['xtcocotools'] = MagicMock()
sys.modules['xtcocotools.coco'] = MagicMock()
