# -*- coding: utf-8 -*-
"""
Created on Wed Apr 11 22:55:45 2018

@author: lsjws2008
"""

from PIL import Image
import numpy as np
from itertools import permutations
from skimage import draw

#def all_zero

def com_loc(cf_map):
    cf = np.copy(cf_map)
    cp = []
    while np.amax(cf) > 0:
        center_point = np.argwhere(cf == np.amax(cf))
        #print(cf.shape)
        cp.append(center_point)
        rd = 1
        if len(center_point.shape) != 2:
            center_point = np.reshape(center_point(1,2))
        #for point in range(center_point.shape(0)):
        #    cf[center_point[point, 0], center_point[point, 1]]
        cf[center_point[:,0],center_point[:,1]] = 0
        #round_rect = list(permutations(list(range(-rd, rd+1)), 2))
        
        arr = np.zeros((200, 200))
        rr, cc = draw.circle_perimeter(100, 100, radius=rd, shape=arr.shape)
        arr[rr, cc] = 1
        round_rect = np.argwhere(arr == 1)-100
        
        rect = []
        for point in range(center_point.shape[0]):
            rect.append(np.array([center_point[point, 0], center_point[point, 1]]) + round_rect)
        rects = np.concatenate(np.array(rect), 0)
        
        while center_point.shape[0] != 0:
            #print(np.amax(rects[:, 0]),np.amax(rects[:, 1]))
            cf[rects[:, 0],rects[:, 1]] = 0
            rd+=1
            #round_rect = list(permutations(list(range(-rd, rd+1)), 2))
            arr = np.zeros((200, 200))
            rr, cc = draw.circle_perimeter(100, 100, radius=rd, shape=arr.shape)
            arr[rr, cc] = 1
            round_rect = np.argwhere(arr == 1)-100
            rect = []
            point = center_point.shape[0]-1
            while point >= 0:
                p_rou = center_point[point].reshape([1,2])
                p_rect = p_rou + round_rect
                p_rect[:, 0] = np.clip(p_rect[:, 0], 0, cf.shape[0]-1)
                p_rect[:, 1] = np.clip(p_rect[:, 1], 0, cf.shape[1]-1)
                #print(round_rect.shape)
                if np.amax(cf[p_rect[:, 0], p_rect[:, 1]]) == 0:
                    
                    center_point = np.delete(center_point, point, axis=0)
                    point-=1
                else:
                    rect.append(p_rect)
                point -= 1
            if rect == []:
                break
            rects = np.concatenate(np.array(rect), 0)
    return np.concatenate(cp, 0)


if __name__ == "__main__":
    root_path = "C:/Users/lsjws2008/Desktop/test_out_img/"
    np.set_printoptions(threshold=np.nan)
    for i in range(17, 33):
        img = Image.open(root_path+str(i)+".jpg")
        confidence_map = np.transpose(np.array(img)[:,:,0], [1, 0])
        confidence_map[confidence_map<170] = 0
        print(com_loc(confidence_map).shape)