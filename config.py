import torch

img_size = 1280
step = [1,2,4,8,12]
'''
#512
img_size = 512
step = [1,2,4,5]
#768
img_size = 768
step = [1,2,4,7]
#1024
img_size = 1024
step = [1,2,4,8,10]
#1280
img_size = 1280
step = [1,2,4,8,12]
#1536
img_size = 1536
step = [1,2,4,8,14]
#1792
img_size = 1792
step = [1,2,4,8,16,17]
#2048
img_size = 2048
step = [1,2,4,8,16,19]
'''

dataset_config = {'root_mut':1,#The multiple of pictures used for training in an epoch
                'img_size':img_size,
                'batch_size':1,#Only 1 avaliable
                'num_workers':1,
                'mosaic':True, 'augment':True,#Whether mocaic and augment method are used for data augmentation
                'degrees':0.0, 'translate':0.1, 'scale':0.5, 'shear': 0.0, 'perspective': 0.0,
                'hsv_h':0.015, 'hsv_s':0.7, 'hsv_v':0.4, 'flipud':0.0, 'fliplr':0.5,
                'for_vis':False,#Set to True only when visualizing the data loader, and generally does not change
                'normalize':True,#Set to False only when visualizing the data loader, and generally does not change
                'gaussian_type':'rectangle',#Whether to use rectangle or square Gaussian kernels, parameters: 'rectangle','square'
                'probmap_radius':2.0#The size control of Gaussian kernels in grape cluster probmap, set to 2 is the same as bbox, less than 2 is greater than bbox, greater than 2 is smaller than bbox
                }
dataset_config_test = {'root_mut':1,#The multiple of pictures used for training in an epoch
                'img_size':img_size,
                'batch_size':1,#Only 1 avaliable
                'num_workers':1,
                'mosaic':False, 'augment':False,#Whether mocaic and augment method are used for data augmentation
                'for_vis':False,#Set to True only when visualizing the data loader, and generally does not change
                'normalize':True,#Set to False only when visualizing the data loader, and generally does not change
                'gaussian_type':'rectangle',#Whether to use rectangle or square Gaussian kernels, parameters: 'rectangle','square'
                'probmap_radius':2.0#The size control of Gaussian kernels in grape cluster probmap, set to 2 is the same as bbox, less than 2 is greater than bbox, greater than 2 is smaller than bbox, generally the same value as the training set
                }

train_config = {'start_epoch':0,
                'epoch':200,
                'ap_epoch':10,#The spoch start using ap, mae and mrd as evaluation metrics
                'lr':2e-4,#Start lr
                'min_lr':1e-6,#End lr
                'schedular':'cosine',#parameters: 'cosine', 'linear'
                'seed':114514
                }

evaluate_config = {'threshold_parts':0.5,#Threshold for detecting berries
                'step':step,#Step of the up-hill method
                'filter':False,#Whether to use cluster probmap to filter berries
                'threshold_filter':0.005,#Threshold used to filter berries with cluster probmap
                'filter_object_score':0,#Keep clusters with a score greater than this value. Set it to 0 to disable the function
                'score_or_multiple':'multiple',#parameters: 'multiple','score'
                'score_alter':0.35,#The threshold used to determine the bbox boundary
                'multiple_alter':2.5,#Used to determine multiples of bbox boundaries
                'min_parts_num':3,#Clusters with berries less than this value will be filtered out
                'vis_type':['img','obj','part']#Visualization parameters: 'train' for training, 'img','obj','part' for evaluation
                }

dev = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
