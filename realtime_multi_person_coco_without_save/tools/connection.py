# -*- coding: utf-8 -*-
"""
Created on Wed Oct 10 20:59:49 2018

@author: johnny
"""

from tools.config_set import config_set
import numpy as np

from numpy import ones,vstack
from numpy.linalg import lstsq

cfg = config_set()

def points_to_line(pA, pB, miu):
    x_coords, y_coords = zip(*[pA, pB])
    x = [pA[0], pB[0]]
    y = [pA[1], pB[1]]
    A = vstack([x_coords,ones(len(x_coords))]).T
    m, c = lstsq(A, y_coords)[0]
     
    x = np.linspace(min(x), max(x)+1, miu)
    y = np.linspace(min(y), max(y)+1, miu)

    if m == 0:
        pass
            
    else:
        x = np.array(x)
        y = m * x + c
    return np.uint(np.stack([x, y], 1))

def compute_line_score(graph, p1, p2, miu):
    discrete_point = points_to_line(p1, p2, miu)
    scores = graph[discrete_point[0, :], discrete_point[0, :]]
    print(scores.shape)
    scores = np.sum(scores)
    return scores

def connection_of_twoPoints(points, line_graph, miu, score_threshold=0.2):
    lines = []
    for ind, i in enumerate(cfg.link_map):
        line = []
        if len(points[i[0]]) == 0 or len(points[i[1]]) == 0:
            continue
        p1 = points[i[0]].tolist()
        p2 = points[i[1]].tolist()
        scores = []
        its = []
        for ij, j in enumerate(p1):
            for ik, k in enumerate(p2):
                its.append([j, k])
                scores.append(compute_line_score(line_graph[..., ind], 
                                                 j, 
                                                 k, 
                                                 miu))
                
        while True:
            max_score = max(scores)
            if max_score < score_threshold:
                break
            ids = scores.index(max_score)
            if its[ids][0] in p1 and its[ids][1] in p2:
                line.append([its[ids][0], its[ids][1]])
                p1.remove(its[ids][0])
                p2.remove(its[ids][1])
            scores.remove(max_score)
            if len(p1) == 0 or len(p2) == 0 or len(scores) == 0:
                break
        lines.append(line)
    return lines