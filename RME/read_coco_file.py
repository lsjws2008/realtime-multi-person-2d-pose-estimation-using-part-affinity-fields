# -*- coding: utf-8 -*-
"""
Created on Thu Jan 18 19:41:18 2018

@author: w835332003
"""
"""
import json
from pprint import pprint
from PIL import Image

def create():
    data = json.load(open('person_keypoints_train2017.json'))
    imgs = []
    labs = []
    
    print(len(data["annotations"]))
    pprint(data["annotations"][0])
    #pprint(data["images"][0])
    Image.open('trains/'+data['images'][0]['file_name'][:-4]+'/img.jpg').show()
    
    for item in data["annotations"]:
        names = "train2017/{:012d}.jpg".format(item['image_id'])
        if names in imgs:
            labs[imgs.index(names)].append(item['keypoints'])
        else:
            imgs.append(names)
            labs.append([item["keypoints"]])
    with open('datas.json','w') as outfile:
        json.dump([imgs,labs], outfile)
    return imgs, labs

"""
from PIL import Image
import numpy as np
import os
import json 

def coco():
    data = json.load(open('datas.json'))
    imgs = data[0]
    labs = data[1]
    count = 0
    for i, l in zip(imgs, labs):
        count += 1
        img = Image.open('trains/'+i[9:-4]+'/img.jpg')
        img_x, img_y = img.size
        for point in range(17):
            point_os = []
            for psi in l:
                if psi[point*3-1] != 0:
                    psit = np.array([psi[point*3-3], psi[point*3-2]])
                    psit[0] = psit[0]*200/img_x
                    psit[1] = psit[1]*100/img_y
                    point_os.append(psit)
                else:
                    point_os.append(np.array([1e8,1e8]))
            point_os = [point_os]*int(400/4)
            point_os = [point_os]*int(800/4)
            point_os = np.array(point_os)
            loc = np.zeros([200,100, len(l), 2])
            for x in range(200):
                for y in range(100):
                    loc[x][y] = [[x,y]]*len(l)
            #print(loc)
            sums = np.sqrt(np.sum(np.power(loc-point_os,2),3))
            #print('\n',np.amin(sums))
            sums = np.amax(np.exp(-(sums/9)), 2)
            sums[sums < 0.3] = 0
            sums = np.expand_dims(sums, 2)
            if point == 0:
                s_lab = sums
            else:
                s_lab = np.concatenate([s_lab, sums], 2)
        fol = 'trains'+i[9:-4]
        print(count, fol)
        if os.path.isfile(fol+'/s_lab.npy'):
            os.remove(fol+'/s_lab.npy')
        np.save(fol+'/s_lab.npy',s_lab)
        img.save(fol+'/img.jpg')
        #print(s_lab.shape)

if __name__ == '__main__':
    coco()