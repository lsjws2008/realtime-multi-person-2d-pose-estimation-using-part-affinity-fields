# -*- coding: utf-8 -*-
"""
Created on Sat Apr 28 17:16:22 2018

@author: lsjws2008
"""

from tools.NMS import com_loc
from tools.PAFs import coms
import numpy as np

def nms(cmap, score_threshold=0.65, score_rate_threshold=0.65):

    points = []
    cmap[cmap< score_threshold] = 0
    for i in range(cmap.shape[2]):
        cm = cmap[:, :, i]
        cm[cm < np.amax(cm)*score_rate_threshold] = 0
        point = com_loc(cm)
        points.append(point)

    return points