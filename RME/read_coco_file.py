# -*- coding: utf-8 -*-
"""
Created on Thu Jan 18 19:41:18 2018

@author: w835332003
"""

"""
psi will problem the kepoint of people shoule be push one point.
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
part = {0:'right ankle', 1:'nose', 2:'left eye', 3:'right eye', 4:'left ear', 5:'right ear',
        6:'left shoulder', 7:'right shoulder', 8:'left elbow', 9:'right elbow', 
        10:'left wrist', 11:'right wrist', 12:'left yao', 13:'right yao', 
        14:'left knee',15:'right knee', 16:'left ankle'}
lines = [[1,2],[1,3],[2,4],[3,5],[6,8],[8,10],[7,9],[9,11],[12,14],[14,16],[13,15],
        [15,0]]
delta_s = 7
delta_l = 4
def coco():
    data = json.load(open('datas.json'))
    imgs = data[0]
    labs = data[1]
    count = 0
    
    for i, l in zip(imgs, labs):
        count += 1
        fol = 'trains'+i[9:-4]
        
        img = Image.open(fol+'/img.jpg')
        img_x, img_y = img.size
        """
        for point in range(17):
            point_os = []
            for psi in l:
                if psi[point*3-1] == 2:
                    psit = np.array([psi[point*3-3], psi[point*3-2]])
                    psit[0] = psit[0]*100/img_x
                    psit[1] = psit[1]*50/img_y
                    point_os.append(psit)
                else:
                    point_os.append(np.array([1e8,1e8]))
            point_os = [point_os]*int(50)
            point_os = [point_os]*int(100)
            point_os = np.array(point_os)
            loc = np.zeros([100,50, len(l), 2])
            for x in range(100):
                for y in range(50):
                    loc[x][y] = [[x,y]]*len(l)
            #print(loc)
            sums = np.sqrt(np.sum(np.power(loc-point_os,2),3))
            #print('\n',np.amin(sums))
            sums = np.amax(np.exp(-(sums/delta_s)), 2)
            sums[sums < 0.3] = 0
            sums = np.expand_dims(sums, 2)
            if point == 0:
                s_lab = sums
            else:
                s_lab = np.concatenate([s_lab, sums], 2)
        
        
        #os.makedirs(fol)
        
        if os.path.isfile(fol+'/s_lab.npy'):
            os.remove(fol+'/s_lab.npy')
        
        np.save(fol+'/s_lab.npy',s_lab)
        #img.save(fol+'/img.jpg')
        #print(s_lab.shape)
        """
        print(count, fol)
        fs = []
        for line in lines:
            
            P = []
            for psi in l:
                #print(psi)       #51
                if psi[line[0]*3-1] == 2 and psi[line[1]*3-1] == 2:
                    P.append([psi[line[0]*3-3]/img_x*100, psi[line[0]*3-2]/img_y*50,psi[line[1]*3-3]/img_x*100, psi[line[1]*3-2]/img_y*50])
            
            if len(P) == 0:
                fs.append(np.zeros([100, 50, 2]))
                continue
            P = np.array(P)
            #print((P[:,1]).reshape([1,1]).shape)
            sh = P.shape[0]
            
            loc = np.zeros([100,50, sh, 2])
            for x in range(100):
                for y in range(50):
                    loc[x][y] = [[x,y]]*sh
                    
            lens = np.sqrt(np.power(P[:,2]-P[:,0], 2)+np.power(P[:,3]-P[:,1], 2))
            lens = lens.reshape(sh, 1)
            V = np.concatenate([(P[:,2]-P[:,0])[..., np.newaxis], (P[:,3]-P[:,1])[..., np.newaxis]],1).reshape([sh, 2])
            #print(P, lens, V)
            V = V/lens
            V_T = np.array([V[:,1], -1*V[:,0]]).reshape([sh,2]) 
            P1 = np.concatenate([P[:,0], P[:,1]],0).reshape(sh,2)
            #V P V_T
            uv = np.copy(V)
            V = [V.tolist()]*int(50)
            V = np.array([V]*int(100))
            
            V_T = [V_T.tolist()]*int(50)
            V_T = np.array([V_T]*int(100))
            
            P1 = [P1.tolist()]*int(50)
            P1 = np.array([P1]*int(100))
            
            di = loc-P1
            
            f1 = di[:,:,:,0]*V[:,:,:,0] + di[:,:,:,1]*V[:,:,:,1]
            f1[f1 < 0] = 0
            for s in range(sh):
                f1[:,:,s][f1[:,:,s] > lens[s]] = 0
            f1[f1 != 0] = 1
            
            f2 = di[:,:,:,0]*V_T[:,:,:,0] + di[:,:,:,1]*V_T[:,:,:,1]
            f2 = np.absolute(f2)
            f2[f2 > delta_l] = 0
            f2[f2 != 0] = 1
            
            fs1 = f1*f2
            fs2 = np.copy(fs1)
            
            fs3 = []
            fs4 = []
            for s in range(sh):
                f = fs1[:,:,s]
                f[f == 1] = uv[s, 0]
                fs3.append(np.expand_dims(f, 2))
                f = fs2[:,:,s]
                f[f == 1] = uv[s, 1]
                fs4.append(np.expand_dims(f, 2))
                
            fs3 = np.concatenate(fs3, 2)
            fs3 = np.sum(fs3, 2)/np.count_nonzero(fs3, 2)
            fs4 = np.concatenate(fs4, 2)
            fs4 = np.sum(fs4, 2)/np.count_nonzero(fs4, 2)
            
            fs5 = np.nan_to_num(np.concatenate([fs3[...,np.newaxis], fs4[...,np.newaxis]], 2))
            #for k in np.nditer(fs5):
            #   print(k)
                
            #print(np.count_nonzero(fs1), np.count_nonzero(fs5))
            fs.append(fs5)
        fs = np.concatenate(fs, 2)
        if np.amax(fs)>1:
            print(np.amax(fs), fol)
            break
        if os.path.isfile(fol+'/l_lab.npy'):
            os.remove(fol+'/l_lab.npy')
        np.save(fol+'/l_lab.npy',fs)
        if count == 50000:
            break

if __name__ == '__main__':
    coco()