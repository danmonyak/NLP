"""
util.py
=======
Author - Daniel Monyak
=======

Module containing utility functions
"""

import numpy as np

def dist2(vec1, vec2):
    return np.sum((vec1 - vec2)**2)


