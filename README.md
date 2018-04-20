# realtime-multi-person-2d-pose-estimation-using-part-affinity-fields  

### This is a project using tensorflow implement realtime-multi-person-2d-pose-estimation-using-part-affinity-fields   
### [paper address](https://arxiv.org/abs/1611.08050)   
### This implement use mpii or COCO dataset.   
  
  
* **conv_net_bn.py** is test or train VGG-net.  
* **load_mpii.py** loading mpii dataset.  
* **read_coco_file.py** loading COCO dataset, and create the label.  
* **trcifar_2.py** use cifar10 pre_train VGG-net.  
* **paxel_net.py** two-branch CNN.  
* **read_npy_pre.py** load pre_trained_VGG-net.[download](https://github.com/machrisaa/tensorflow-vgg)  
* **train_coco.py** train CNN use COCO dataset.  
* **test_mpii.py** test the CNN.  
* **NMS.py** catch the keypint from confidence map. Similar ***non maximum suppression***
