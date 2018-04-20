from load_mpii import generate_file
from random import shuffle
from PIL import Image
import numpy as np
import config_set
from math import exp

class mpii():
    
    def __init__(self, mode):
        
        np.set_printoptions(threshold=np.nan)
        self.config = config_set.config_set()
        self.link_map = self.config.link_map
        self.data = generate_file()
        self.mode = mode
        self.rem()
        if self.mode == 'train':
            self.first_line()
        elif self.mode == 'test':
            pass
        else:
            self.first_line()
        self.ind = list(range(len(self.data)))
        shuffle(self.ind)
    def rem(self):
        #rm = []
        for i in self.data:
            if i['persons_anno'] == 0:
                self.data.remove(i)

    def first_line(self):
        data = []
        for i in range(len(self.data)):
            if self.data[i]['seq'] == 1:
                data.append(self.data[i])
        self.data = data
    
    def in_link(p, v):
        pass
    
    def batch_data(self, batch):
        if len(self.ind) < batch:
            self.ind = list(range(len(self.data)))
            shuffle(self.ind)
        imgs = []
        s_labs = []
        s_com_labs = []
        #l_labs = []
        for i in range(batch):
            struct = self.data[self.ind[0]]
            img = Image.open('images/'+struct['filename'])
            img_x, img_y = img.size
            img = img.resize((self.config.img_x, self.config.img_y))
            imgs.append(np.transpose(np.array(img), [1, 0, 2])/255)
            s_lab = []
            s_com_lab = []
            for point in range(self.config.point_num):
                point_os = []
                for psi in struct['persons_anno']:
                    if point in psi:
                        psi = np.array([psi[psi.index(point)-2], psi[psi.index(point)-1]])
                        psi[0] = psi[0]*200/img_x
                        psi[1] = psi[1]*100/img_y
                        point_os.append(psi)
                    else:
                        point_os.append(np.array([1e8,1e8]))
                point_os = [point_os]*int(400/self.config.under_pool)
                point_os = [point_os]*int(800/self.config.under_pool)
                point_os = np.array(point_os)
                loc = np.zeros([200,100, len(struct['persons_anno']), 2])
                for x in range(200):
                    for y in range(100):
                        loc[x][y] = [[x,y]]*len(struct['persons_anno'])
                
                #print(loc)
                sums = np.sqrt(np.sum(np.power(loc-point_os,2),3))
                #print('\n',np.amin(sums))
                sums = np.amax(np.exp(-(sums/self.config.paxul)), 2)
                sums[sums < 0.35] = 0
                count = np.count_nonzero(sums)
                com_lab = np.copy(sums)
                com_lab[com_lab>=0.1]=1
                x = list(range(200))
                y = list(range(100))
                while(count>0):
                    loc_x = np.random.choice(x,1)
                    loc_y = np.random.choice(y,1)
                    if com_lab[loc_x, loc_y] == 0:
                        com_lab[loc_x, loc_y] = 1
                        count-=1
                if len(s_lab) == 0:
                    s_com_lab = np.expand_dims(com_lab, 2)
                    s_lab  = np.expand_dims(sums, 2)
                else:
                    s_lab = np.concatenate((s_lab, np.expand_dims(sums, 2)), 2)
                    s_com_lab = np.concatenate((s_com_lab, np.expand_dims(com_lab, 2)), 2)
            s_labs.append(s_lab)
            s_com_labs.append(s_com_lab)
            self.ind.remove(self.ind[0])
            
        print(np.mean(s_labs))
        return imgs, s_labs, s_com_labs
if __name__ == '__main__':
    dataset = mpii('train')
    k = dataset.batch_data(1)
    s = k[1][0]
    for i in range(16):
        map_0 = s[:, :, i]
        print(np.count_nonzero(map_0))
        map_0 = np.expand_dims(map_0, 2)
        map_0 = map_0*255
        map_0_to = np.concatenate((map_0, map_0), 2)
        map_0_to = np.concatenate((map_0_to, map_0), 2)
        im = Image.fromarray(np.uint8(map_0_to).transpose((1, 0, 2)))
        #im.show()
        im.save('test_out_img/'+str(i)+'.jpg')
    s = k[2][0]
    for i in range(16):
        map_0 = s[:, :, i]
        print(np.count_nonzero(map_0))
        map_0 = np.expand_dims(map_0, 2)
        map_0 = map_0*255
        map_0_to = np.concatenate((map_0, map_0), 2)
        map_0_to = np.concatenate((map_0_to, map_0), 2)
        im = Image.fromarray(np.uint8(map_0_to).transpose((1, 0, 2)))
        #im.show()
        im.save('test_out_img/'+str(i+16)+'.jpg')
    print(np.array(s).shape)