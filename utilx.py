import sys
import cv2
import numpy as np
from torch import torch, utils, nn
import yaml


def blank_frame(rgb_color, target_dim, configx=None):
    image = np.zeros((target_dim[1], target_dim[0], 3), np.uint8)
    image[:] = tuple(rgb_color)
    return image


def swap_RGB2BGR(matrix):
    red = matrix[:,:,0].copy()
    blue = matrix[:,:,2].copy()
    matrix[:,:,0] = blue
    matrix[:,:,2] = red
    return matrix


class AverageMeter(object):
    def __init__(self):
        self.reset()
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class IOUScore(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, Yp, Yt):
        output_ = Yp > 0.5 
        target_ = Yt > 0.5 
        intersection = (output_ & target_).sum() 
        union = (output_ | target_).sum() 
        iou = intersection / union
        return iou

class BCEDiceLoss(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, Yp, Yt, smooth=1e-7):
        num = Yt.size(0)
        Yp = Yp.view(num, -1)
        Yt = Yt.view(num, -1)
        bce = nn.functional.binary_cross_entropy(Yp, Yt)
        intersection = (Yp * Yt).sum() 
        dice_loss = 1 - ((2. * intersection + smooth) / (Yp.sum() + Yt.sum() + smooth))
        bce_dice_loss = bce + dice_loss
        return bce_dice_loss


class L1Loss(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, Yp, Yt):
        num = Yt.size(0)
        Yp = Yp.view(num, -1)
        Yt = Yt.view(num, -1)
        loss = nn.functional.l1_loss(Yp, Yt)
        return loss

class HuberLoss(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, Yp, Yt):
        num = Yt.size(0)
        Yp = Yp.view(num, -1)
        Yt = Yt.view(num, -1)
        loss = nn.functional.smooth_l1_loss(Yp, Yt, beta=0.5)

        return loss


class datagen(utils.data.Dataset):
    def __init__(self, file_ids, config, data_info, input_dir):
        self.file_ids = file_ids
        self.config = config
        self.data_info = data_info
        self.input_dir = input_dir

    def __len__(self):
        return len(self.file_ids)

    def __getitem__(self, idx):
        file_id = self.file_ids[idx]
        ddir = self.input_dir

        #DVS
        dvs_f = cv2.imread(ddir+self.config['input'][0]+"/"+file_id) / 255
        dvs_l = cv2.imread(ddir+self.config['input'][1]+"/"+file_id) / 255
        dvs_ri = cv2.imread(ddir+self.config['input'][2]+"/"+file_id) / 255
        dvs_r = cv2.imread(ddir+self.config['input'][3]+"/"+file_id) / 255
        dvs_f = np.delete(dvs_f.transpose(2, 0, 1), obj=1, axis=0)
        dvs_l = np.delete(dvs_l.transpose(2, 0, 1), obj=1, axis=0)
        dvs_ri = np.delete(dvs_ri.transpose(2, 0, 1), obj=1, axis=0)
        dvs_r = np.delete(dvs_r.transpose(2, 0, 1), obj=1, axis=0)

        #BGR
        bgr_f = cv2.imread(ddir+self.config['input'][4]+"/"+file_id) / 25
        bgr_l = cv2.imread(ddir+self.config['input'][5]+"/"+file_id) / 255
        bgr_ri = cv2.imread(ddir+self.config['input'][6]+"/"+file_id) / 255
        bgr_r = cv2.imread(ddir+self.config['input'][7]+"/"+file_id) / 255
        bgr_f = bgr_f.transpose(2, 0, 1)
        bgr_l = bgr_l.transpose(2, 0, 1)
        bgr_ri = bgr_ri.transpose(2, 0, 1)
        bgr_r = bgr_r.transpose(2, 0, 1)

        #input
        inp = (dvs_f, dvs_l, dvs_ri, dvs_r, bgr_f, bgr_l, bgr_ri, bgr_r)

        #DE
        depth_f = cv2.imread(ddir+self.config['task'][0]+"/"+file_id, cv2.COLOR_BGR2GRAY) / 255
        depth_l = cv2.imread(ddir+self.config['task'][1]+"/"+file_id, cv2.COLOR_BGR2GRAY) / 255
        depth_ri = cv2.imread(ddir+self.config['task'][2]+"/"+file_id, cv2.COLOR_BGR2GRAY) / 255
        depth_r = cv2.imread(ddir+self.config['task'][3]+"/"+file_id, cv2.COLOR_BGR2GRAY) / 255
        depth_f = np.expand_dims(depth_f, axis=0)
        depth_l = np.expand_dims(depth_l, axis=0)
        depth_ri = np.expand_dims(depth_ri, axis=0)
        depth_r = np.expand_dims(depth_r, axis=0)

        #SS
        segmentation_f = np.load(ddir+self.config['task'][4]+"/"+file_id[:-4]+"_128.npy")
        segmentation_l = np.load(ddir+self.config['task'][5]+"/"+file_id[:-4]+"_128.npy")
        segmentation_ri = np.load(ddir+self.config['task'][6]+"/"+file_id[:-4]+"_128.npy")
        segmentation_r = np.load(ddir+self.config['task'][7]+"/"+file_id[:-4]+"_128.npy")

        #output                  
        out = (depth_f, depth_l, depth_ri, depth_r, segmentation_f, segmentation_l, segmentation_ri, segmentation_r)

        return inp, out, {'img_id': file_id}



#UNTUK W-UNET train0wunet.py
class datagen0wunet(utils.data.Dataset):
    def __init__(self, file_ids, config, input_dir):#, mode='training', data_dir, inputd, task, aug_transform):  data_info
        self.file_ids = file_ids
        #self.data_dir = data_dir
        #self.n_class = n_class
        #self.inputd = inputd
        #self.task = task
        self.config = config
        # self.data_info = data_info
        #self.mode = mode
        self.input_dir = input_dir
        # self.dev = dev

    def __len__(self):
        return len(self.file_ids)

    def __getitem__(self, idx):
        file_id = self.file_ids[idx] #Index yang dibaca
        #print(file_id)
        #file_id_rgb = file_id[:-3]+"jpg" #jpg png
        ddir = self.input_dir
        #ddirx = '/media/oskar/data/oskar/CARLA/autonomous-vehicle_COPY-20210630_ACPR2021_SUBMIT/'+ddir[:14]
        #INPUT
        #BACA input RGB, normalisasi, resize, dan transpose channel first
        bgr = cv2.imread(ddir+self.config['input'][0]+"/"+file_id) / 255#+".jpg")
        #input diperbesar 4x dulu karena outputnya akan 4x lebih kecil
        #bgr = cv2.resize(bgr, (self.config['tensor_dim'][3], self.config['tensor_dim'][2]), interpolation=cv2.INTER_AREA)
        bgr = bgr.transpose(2, 0, 1)
        # print(bgr.shape)
        #buat list input
        inp = bgr
        


        #OUTPUT
        #baca GT DEPTH gray, normalisasi, resize, dan expand dim channel first
        depth = cv2.imread(ddir+self.config['task'][0]+"/"+file_id, cv2.COLOR_BGR2GRAY) / 255
        # depth = cv2.resize(depth, (self.config['tensor_dim'][3], self.config['tensor_dim'][2]), interpolation=cv2.INTER_AREA)
        depth = np.expand_dims(depth, axis=0)
        # print(depth.shape)

        #SEGMENTATION
        # if self.data_info['n_seg_class'] == 23:
        segmentation = np.load(ddir+self.config['task'][1]+"/"+file_id[:-4]+"_128.npy")
        # else:
        #     segmentation_f = np.load(ddir+self.config['task'][4]+"/"+file_id[:-4]+"_128_min.npy")
        #     segmentation_l = np.load(ddir+self.config['task'][5]+"/"+file_id[:-4]+"_128_min.npy")
        #     segmentation_ri = np.load(ddir+self.config['task'][6]+"/"+file_id[:-4]+"_128_min.npy")
        #     segmentation_r = np.load(ddir+self.config['task'][7]+"/"+file_id[:-4]+"_128_min.npy")
        #jadikan ke tensor
        # segmentation_f = torch.tensor(segmentation_f, dtype=torch.double, device=self.dev)
        # segmentation_l = torch.tensor(segmentation_l, dtype=torch.double, device=self.dev)
        # segmentation_ri = torch.tensor(segmentation_ri, dtype=torch.double, device=self.dev)
        # segmentation_r = torch.tensor(segmentation_r, dtype=torch.double, device=self.dev)
                #else: #berarti pakai yang hanya 13 class
        #    segmentation_f = np.load(ddir+self.config['task'][4]+"/"+file_id[:-4]+"_128_min.npy")
        #    segmentation_r = np.load(ddir+self.config['task'][5]+"/"+file_id[:-4]+"_128_min.npy")
    
        #buat list output                  
        out = (depth, segmentation)
        #else:
        #out = (depth_f, depth_l, depth_ri, depth_r,
        #        segmentation_f, segmentation_l, segmentation_ri, segmentation_r,
        #        bird_view)

        
        return inp, out, {'img_id': file_id}



#ini khusus untuk predict0wunet_all.py
class datagen0wunet_all(utils.data.Dataset):
    def __init__(self, file_ids, config0, config1, config2, config3, input_dir):#, mode='training', data_dir, inputd, task, aug_transform): 
        self.file_ids = file_ids
        #self.data_dir = data_dir
        #self.n_class = n_class
        #self.inputd = inputd
        #self.task = task
        self.config0 = config0
        self.config1 = config1
        self.config2 = config2
        self.config3 = config3
        #self.mode = mode
        self.input_dir = input_dir
        # self.dev = dev

    def __len__(self):
        return len(self.file_ids)

    def __getitem__(self, idx):
        file_id = self.file_ids[idx] #Index yang dibaca
        
        ddir = self.input_dir

        #INPUT
        #BACA input RGB, normalisasi, resize, dan transpose channel first
        bgr_f = cv2.imread(ddir+self.config0['input'][0]+"/"+file_id) / 255#+".jpg")
        bgr_l = cv2.imread(ddir+self.config1['input'][0]+"/"+file_id) / 255
        bgr_ri = cv2.imread(ddir+self.config2['input'][0]+"/"+file_id) / 255
        bgr_r = cv2.imread(ddir+self.config3['input'][0]+"/"+file_id) / 255
        # bgr_f = cv2.resize(bgr_f, (self.config['tensor_dim'][3], self.config['tensor_dim'][2]), interpolation=cv2.INTER_AREA)
        # bgr_l = cv2.resize(bgr_l, (self.config['tensor_dim'][3], self.config['tensor_dim'][2]), interpolation=cv2.INTER_AREA)
        # bgr_ri = cv2.resize(bgr_ri, (self.config['tensor_dim'][3], self.config['tensor_dim'][2]), interpolation=cv2.INTER_AREA)
        # bgr_r = cv2.resize(bgr_r, (self.config['tensor_dim'][3], self.config['tensor_dim'][2]), interpolation=cv2.INTER_AREA)
        bgr_f = bgr_f.transpose(2, 0, 1)
        bgr_l = bgr_l.transpose(2, 0, 1)
        bgr_ri = bgr_ri.transpose(2, 0, 1)
        bgr_r = bgr_r.transpose(2, 0, 1)
        #jadikan ke tensor
        # bgr_f = torch.tensor(bgr_f, dtype=torch.double, device=self.dev)
        # bgr_l = torch.tensor(bgr_l, dtype=torch.double, device=self.dev)
        # bgr_ri = torch.tensor(bgr_ri, dtype=torch.double, device=self.dev)
        # bgr_r = torch.tensor(bgr_r, dtype=torch.double, device=self.dev)

        #buat list input
        inp = (bgr_f, bgr_l, bgr_ri, bgr_r)
        


        #OUTPUT
        #baca GT DEPTH gray, normalisasi, resize, dan expand dim channel first
        depth_f = cv2.imread(ddir+self.config0['task'][0]+"/"+file_id, cv2.COLOR_BGR2GRAY) / 255
        depth_l = cv2.imread(ddir+self.config1['task'][0]+"/"+file_id, cv2.COLOR_BGR2GRAY) / 255
        depth_ri = cv2.imread(ddir+self.config2['task'][0]+"/"+file_id, cv2.COLOR_BGR2GRAY) / 255
        depth_r = cv2.imread(ddir+self.config3['task'][0]+"/"+file_id, cv2.COLOR_BGR2GRAY) / 255
        # depth_f = cv2.resize(depth_f, (self.config['tensor_dim'][3], self.config['tensor_dim'][2]), interpolation=cv2.INTER_AREA)
        # depth_l = cv2.resize(depth_l, (self.config['tensor_dim'][3], self.config['tensor_dim'][2]), interpolation=cv2.INTER_AREA)
        # depth_ri = cv2.resize(depth_ri, (self.config['tensor_dim'][3], self.config['tensor_dim'][2]), interpolation=cv2.INTER_AREA)
        # depth_r = cv2.resize(depth_r, (self.config['tensor_dim'][3], self.config['tensor_dim'][2]), interpolation=cv2.INTER_AREA)
        depth_f = np.expand_dims(depth_f, axis=0)
        depth_l = np.expand_dims(depth_l, axis=0)
        depth_ri = np.expand_dims(depth_ri, axis=0)
        depth_r = np.expand_dims(depth_r, axis=0)
        #jadikan ke tensor
        # depth_f = torch.tensor(depth_f, dtype=torch.double, device=self.dev)
        # depth_l = torch.tensor(depth_l, dtype=torch.double, device=self.dev)
        # depth_ri = torch.tensor(depth_ri, dtype=torch.double, device=self.dev)
        # depth_r = torch.tensor(depth_r, dtype=torch.double, device=self.dev)


        #SEGMENTATION
        # if self.data_info['n_seg_class'] == 23:
        segmentation_f = np.load(ddir+self.config0['task'][1]+"/"+file_id[:-4]+"_128.npy")
        segmentation_l = np.load(ddir+self.config1['task'][1]+"/"+file_id[:-4]+"_128.npy")
        segmentation_ri = np.load(ddir+self.config2['task'][1]+"/"+file_id[:-4]+"_128.npy")
        segmentation_r = np.load(ddir+self.config3['task'][1]+"/"+file_id[:-4]+"_128.npy")
        # else:
        #     segmentation_f = np.load(ddir+self.config['task'][4]+"/"+file_id[:-4]+"_128_min.npy")
        #     segmentation_l = np.load(ddir+self.config['task'][5]+"/"+file_id[:-4]+"_128_min.npy")
        #     segmentation_ri = np.load(ddir+self.config['task'][6]+"/"+file_id[:-4]+"_128_min.npy")
        #     segmentation_r = np.load(ddir+self.config['task'][7]+"/"+file_id[:-4]+"_128_min.npy")
        #jadikan ke tensor
        # segmentation_f = torch.tensor(segmentation_f, dtype=torch.double, device=self.dev)
        # segmentation_l = torch.tensor(segmentation_l, dtype=torch.double, device=self.dev)
        # segmentation_ri = torch.tensor(segmentation_ri, dtype=torch.double, device=self.dev)
        # segmentation_r = torch.tensor(segmentation_r, dtype=torch.double, device=self.dev)
                #else: #berarti pakai yang hanya 13 class
        #    segmentation_f = np.load(ddir+self.config['task'][4]+"/"+file_id[:-4]+"_128_min.npy")
        #    segmentation_r = np.load(ddir+self.config['task'][5]+"/"+file_id[:-4]+"_128_min.npy")
    
        #buat list output                  
        out = (depth_f, depth_l, depth_ri, depth_r,
                segmentation_f, segmentation_l, segmentation_ri, segmentation_r)
        #else:
        #out = (depth_f, depth_l, depth_ri, depth_r,
        #        segmentation_f, segmentation_l, segmentation_ri, segmentation_r,
        #        bird_view)

        
        return inp, out, {'img_id': file_id}










