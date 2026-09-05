import re
with open("wholebody/structures/keypoint_spec.py", "r") as f:
    text = f.read()

# I will just write a python script that doesn't import any heavy libraries
import ast
import imp
import sys
# Actually just parse with a regex
import re

# ... just look at the python script
