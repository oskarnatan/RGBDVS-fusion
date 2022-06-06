import cv2
import numpy as np
from torch import utils, nn


def blank_frame(rgb_color, target_dim):
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


