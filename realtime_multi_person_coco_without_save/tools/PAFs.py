# -*- coding: utf-8 -*-
"""
Created on Tue Apr 17 21:00:38 2018

@author: lsjws2008
"""

import numpy as np

def com_L(point_p, points, field, us = 10):
    if len(points.shape) == 1:
        points = np.reshape(points, [2, 1])
    if len(point_p.shape) == 1:
        point_p = np.reshape(point_p, [2, 1])
    point_v = point_p - points
    point_l = np.sqrt(np.sum(np.power(point_v, 2) ,0))
    point_u = np.nan_to_num(point_v/point_l)
    ls = []
    for i in range(us+1):
        miu = i/us
        point_f = np.int32(np.round(point_p*(1-miu) + points*miu))
        px = field[point_f[0, :], point_f[1, :], [0]*points.shape[1]]
        py = field[point_f[0, :], point_f[1, :], [1]*points.shape[1]]
        l = []
        for j in range(points.shape[1]):
            #print(point_v)
            #print(np.dot(np.array([px,py])[:, j], point_u[:, j]))
            l.append(np.dot(np.array([px,py])[:, j], point_u[:, j]))
        ls.append(l)
    ls = np.sum(np.array(ls), 0)
    return ls

def coms(points_p, points, field, us = 5):
    points_p = np.transpose(points_p)
    points = np.transpose(points)
    if len(points_p.shape) == 1:
        points_p = np.reshape(points_p, (2,1))
    if len(points.shape) == 1:
        points = np.reshape(points, (2,1))
    lines = []
    c = points.shape[1]
    ls = []
    for i in range(c):
        arg = com_L(points_p, points[:, i], field, us)
        ls.append(arg)
        #print(arg.shape)
        #if np.amax(arg) > 0:
            #lines.append([points_p[:, np.argwhere(arg == np.amax(arg))[0][0]], points[:, i]])
            #points_p = np.delete(points_p, arg, 1)
            #break
    ls = np.array(ls)
    print(ls)
    while np.amax(ls) > 0 :
        line_psi = np.argwhere(ls == np.amax(ls))
        if line_psi.shape[1] != 1:
            lines.append([points_p[:,line_psi[0, 1]], points[:,line_psi[0, 0]]])
            ls = np.delete(ls, line_psi)
        else:
            break
        if ls.shape[0] == 0:
            break
    #while 0 in points.shape or 0 in points_p.shape:    
    return lines

if __name__ == "__main__":
    
    point_p = np.ones([8,2])
    points = range(4)
    points = np.reshape(points, [2,2]) + 1e-3
    field = np.ones([100,50,2])
    print(coms(point_p, points, field, 5))