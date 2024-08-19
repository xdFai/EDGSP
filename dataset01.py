# -*- coding: utf-8 -*-
# @Author  : Shuai Yuan
# @File    : dataset01.py
# @Software: PyCharm
# coding=utf-8
from utils import *
from torch.utils.data import Dataset
import os
from skimage import measure

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


# ********************************************************
#      Gaussian    mode: centroid  or coarse   aug: True or False
#      input:   image  +  point_Gaussian
# ********************************************************
class TrainSetLoader_Gauss(Dataset):
    def __init__(self, dataset_dir, dataset_name, patch_size, sigma, train_mode, aug, img_norm_cfg=None):
        super(TrainSetLoader_Gauss).__init__()
        self.dataset_name = dataset_name
        self.dataset_dir = dataset_dir + '/' + dataset_name
        self.patch_size = patch_size
        self.sigma = sigma
        self.train_mode = train_mode
        self.aug = aug
        with open(self.dataset_dir + '/img_idx/train_' + dataset_name + '.txt', 'r') as f:
            self.train_list = f.read().splitlines()
        if img_norm_cfg == None:
            self.img_norm_cfg = get_img_norm_cfg(dataset_name, dataset_dir)
        else:
            self.img_norm_cfg = img_norm_cfg
        self.tranform_Pce = augumentation_Pce()

    def __getitem__(self, idx):
        img = Image.open((self.dataset_dir + '/images/' + self.train_list[idx] + '.png').replace('//', '/')).convert(
            'I')
        mask = Image.open((self.dataset_dir + '/masks/' + self.train_list[idx] + '.png').replace('//', '/'))
        if self.train_mode == 'centroid':
            point = Image.open((self.dataset_dir + '/Centroid/' + self.train_list[idx] + '.png').replace('//', '/'))
        elif self.train_mode == 'coarse':
            point = Image.open((self.dataset_dir + '/masks_coarse/' + self.train_list[idx] + '.png').replace('//', '/'))
        # ****************** convert PIL to numpy  and  normalize ******************
        img_patch = Normalized(np.array(img, dtype=np.float32), self.img_norm_cfg)
        mask_patch = np.array(mask, dtype=np.float32) / 255.0
        point_input = np.array(point, dtype=np.float32) / 255.0

        img_patch, mask_patch, point_input = random_crop_Pce(img_patch, mask_patch, point_input, self.patch_size,
                                                             pos_prob=0.5)
        img_patch, mask_patch, point_input = self.tranform_Pce(img_patch, mask_patch, point_input)  # 翻转增强

        if len(mask_patch.shape) > 2:
            mask = mask_patch[:, :, 0]
            point = point_input[:, :, 0]


        # **************************************************************************
        #                 Target Energy Initialization (TEI)
        # **************************************************************************
        if self.sigma > 0:
            #  Change data type
            mask_p = measure.label(point_input, connectivity=2)
            coord_m = measure.regionprops(mask_p)
            point_input_p = np.zeros(shape=(self.patch_size, self.patch_size), dtype=np.float32)
            for i in range(len(coord_m)):
                aa = np.array(list(coord_m[i].centroid))
                x = round(aa[0])
                y = round(aa[1])
                point_input_p[x, y] = 1
            point_input = Gaussian_dis(point_input_p, self.sigma)
        elif self.sigma == 0:
            #  Change data type
            mask_p = measure.label(point_input, connectivity=2)
            coord_m = measure.regionprops(mask_p)
            point_input_p = np.zeros(shape=(self.patch_size, self.patch_size), dtype=np.float32)
            for i in range(len(coord_m)):
                aa = np.array(list(coord_m[i].centroid))
                x = round(aa[0])
                y = round(aa[1])
                point_input_p[x, y] = 1
            point_input_p = point_input_p[np.newaxis, :]  # 升维
            point_input = torch.from_numpy(np.ascontiguousarray(point_input_p))  # numpy 转tensor
        else:
            point_input = point_input[np.newaxis, :]
            point_input = torch.from_numpy(np.ascontiguousarray(point_input))

        # ************** increase dim, numpy->tensor ***************************
        img_patch, mask_patch = img_patch[np.newaxis, :], mask_patch[np.newaxis, :]
        img_patch = torch.from_numpy(np.ascontiguousarray(img_patch))
        mask_patch = torch.from_numpy(np.ascontiguousarray(mask_patch))

        return img_patch, mask_patch, point_input

    def __len__(self):
        return len(self.train_list)


# ********************************************************
#      Gaussian  mode: centroid  or  coarse
#      input:   image  +  point_Gaussian
# ********************************************************
class TestSetLoader_Gauss(Dataset):
    def __init__(self, dataset_dir, train_dataset_name, test_dataset_name, sigma, test_mode, img_norm_cfg=None):
        super(TestSetLoader_Gauss).__init__()
        self.dataset_dir = dataset_dir + '/' + test_dataset_name
        self.sigma = sigma
        self.test_mode = test_mode
        with open(self.dataset_dir + '/img_idx/test_' + test_dataset_name + '.txt', 'r') as f:
            self.test_list = f.read().splitlines()
        if img_norm_cfg == None:
            self.img_norm_cfg = get_img_norm_cfg(train_dataset_name, dataset_dir)
        else:
            self.img_norm_cfg = img_norm_cfg

    def __getitem__(self, idx):
        img = Image.open((self.dataset_dir + '/images/' + self.test_list[idx] + '.png').replace('//', '/')).convert('I')
        mask = Image.open((self.dataset_dir + '/masks/' + self.test_list[idx] + '.png').replace('//', '/'))
        if self.test_mode == 'centroid':
            point = Image.open((self.dataset_dir + '/Centroid/' + self.test_list[idx] + '.png').replace('//', '/'))
        elif self.test_mode == 'U04':
            point = Image.open((self.dataset_dir + '/U04/' + self.test_list[idx] + '.png').replace('//', '/'))
        elif self.test_mode == 'U04_Silence':
            point = Image.open((self.dataset_dir + '/U04_Silence/' + self.test_list[idx] + '.png').replace('//', '/'))
        elif self.test_mode == 'coarse':
            point = Image.open((self.dataset_dir + '/masks_coarse/' + self.test_list[idx] + '.png').replace('//', '/'))

        img = Normalized(np.array(img, dtype=np.float32), self.img_norm_cfg)
        mask = np.array(mask, dtype=np.float32) / 255.0
        point = np.array(point, dtype=np.float32) / 255.0
        # if mask.shape == (416,608):
        #     print('111')
        if len(mask.shape) > 2:
            mask = mask[:, :, 0]
            point = point[:, :, 0]

        h, w = img.shape
        img = PadImg(img)
        mask = PadImg(mask)
        point_input = PadImg(point)
        k, b = img.shape
        # **************************************************************************
        #                 Target Energy Initialization (TEI)
        # **************************************************************************
        if self.sigma > 0:
            mask_p = measure.label(point_input, connectivity=2)
            coord_m = measure.regionprops(mask_p)
            point_input_p = np.zeros(shape=(k, b), dtype=np.float32)
            for i in range(len(coord_m)):
                aa = np.array(list(coord_m[i].centroid))
                x = round(aa[0])
                y = round(aa[1])
                point_input_p[x, y] = 1
            point_input = Gaussian_dis(point_input_p, self.sigma)
        elif self.sigma == 0:
            mask_p = measure.label(point_input, connectivity=2)
            coord_m = measure.regionprops(mask_p)
            point_input_p = np.zeros(shape=(k, b), dtype=np.float32)
            for i in range(len(coord_m)):
                aa = np.array(list(coord_m[i].centroid))
                x = round(aa[0])
                y = round(aa[1])
                point_input_p[x, y] = 1
            point_input_p = point_input_p[np.newaxis, :]  # 升维
            point_input = torch.from_numpy(np.ascontiguousarray(point_input_p))  # numpy 转tensor
        else:
            point_input = point_input[np.newaxis, :]
            point_input = torch.from_numpy(np.ascontiguousarray(point_input))

        img_patch, mask_patch = img[np.newaxis, :], mask[np.newaxis, :]

        # ************** increase dim, numpy->tensor ***************************
        img_patch = torch.from_numpy(np.ascontiguousarray(img_patch))
        mask_patch = torch.from_numpy(np.ascontiguousarray(mask_patch))
        if img_patch.size() != mask_patch.size():
            print('111')
        return img_patch, mask_patch, point_input, [h, w], self.test_list[idx]

    def __len__(self):
        return len(self.test_list)




class augumentation_Pce(object):
    def __call__(self, input, target, target_Pce):
        if random.random() < 0.5:  
            input = input[::-1, :]
            target = target[::-1, :]
            target_Pce = target_Pce[::-1, :]
        if random.random() < 0.5: 
            input = input[:, ::-1]
            target = target[:, ::-1]
            target_Pce = target_Pce[:, ::-1]
        if random.random() < 0.5:  
            input = input.transpose(1, 0)
            target = target.transpose(1, 0)
            target_Pce = target_Pce.transpose(1, 0)
        return input, target, target_Pce




def Gaussian_dis(point_label, sigma):
    # *************************** Gaussian ***************************
    if np.max(point_label) != 0:
        _points = single_point(point_label)
        point_Gaussian = make_gt(point_label, _points, sigma=sigma, one_mask_per_point=False)
    else:
        point_Gaussian = point_label
    point_Gaussian = point_Gaussian[np.newaxis, :]  # 升维
    point_label = torch.from_numpy(np.ascontiguousarray(point_Gaussian))  # numpy 转tensor

    return point_label


def make_gaussian(size, sigma=10, center=None, d_type=np.float32):
    """ Make a square gaussian kernel.
    size: is the dimensions of the output gaussian
    sigma: is full-width-half-maximum, which
    can be thought of as an effective radius.
    """

    x = np.arange(0, size[1], 1, float)
    y = np.arange(0, size[0], 1, float)
    y = y[:, np.newaxis]

    if center is None:
        x0 = y0 = size[0] // 2
    else:
        x0 = center[0]
        y0 = center[1]

    return np.exp(-4 * np.log(2) * ((x - x0) ** 2 + (y - y0) ** 2) / sigma ** 2).astype(d_type)


def make_gt(img, labels, sigma=10, one_mask_per_point=False):
    """ Make the ground-truth for  landmark.
    img: the original color image
    labels: label with the Gaussian center(s) [[x0, y0],[x1, y1],...]
    sigma: sigma of the Gaussian.
    one_mask_per_point: masks for each point in different channels?
    """
    h, w = img.shape[:2]
    if labels is None:
        gt = make_gaussian((h, w), center=(h // 2, w // 2), sigma=sigma)
    else:
        labels = np.array(labels)
        if labels.ndim == 1:
            labels = labels[np.newaxis]
        if one_mask_per_point:
            gt = np.zeros(shape=(h, w, labels.shape[0]))
            for ii in range(labels.shape[0]):
                gt[:, :, ii] = make_gaussian((h, w), center=labels[ii, :], sigma=sigma)
        else:
            gt = np.zeros(shape=(h, w), dtype=np.float32)
            for ii in range(labels.shape[0]):
                gt = np.maximum(gt, make_gaussian((h, w), center=labels[ii, :], sigma=sigma))

    gt = gt.astype(dtype=img.dtype)

    return gt


def single_point(mask):
    inds_y, inds_x = np.where(mask > 0.5)
    _points_local = []
    for i in range(inds_y.size):
        _points_local.append([inds_x[i], inds_y[i]])

    return np.array(_points_local)




class TestSetLoader_Gauss_for_PdFa(Dataset):
    def __init__(self, dataset_dir, train_dataset_name, test_dataset_name, sigma, test_mode, PF_mode,
                 img_norm_cfg=None):
        super(TestSetLoader_Gauss_for_PdFa).__init__()
        self.dataset_dir = dataset_dir + '/' + test_dataset_name
        self.sigma = sigma
        self.test_mode = test_mode
        self.Pd_fa = PF_mode
        with open(self.dataset_dir + '/img_idx/test_' + test_dataset_name + '.txt', 'r') as f:
            self.test_list = f.read().splitlines()
        if img_norm_cfg == None:
            self.img_norm_cfg = get_img_norm_cfg(train_dataset_name, dataset_dir)
        else:
            self.img_norm_cfg = img_norm_cfg

    def __getitem__(self, idx):
        img = Image.open((self.dataset_dir + '/images/' + self.test_list[idx] + '.png').replace('//', '/')).convert('I')
        mask = Image.open((self.dataset_dir + '/masks/' + self.test_list[idx] + '.png').replace('//', '/'))

        if self.test_mode == 'centroid':
            point = Image.open((self.dataset_dir + '/Centroid/' + self.test_list[idx] + '.png').replace('//', '/'))
        elif self.test_mode == 'coarse':
            point = Image.open((self.dataset_dir + '/masks_coarse/' + self.test_list[idx] + '.png').replace('//', '/'))

        if self.Pd_fa == 'centroid':
            PF_point = Image.open((self.dataset_dir + '/Centroid/' + self.test_list[idx] + '.png').replace('//', '/'))
        elif self.Pd_fa == 'coarse':
            PF_point = Image.open(
                (self.dataset_dir + '/masks_coarse/' + self.test_list[idx] + '.png').replace('//', '/'))

        img = Normalized(np.array(img, dtype=np.float32), self.img_norm_cfg)
        mask = np.array(mask, dtype=np.float32) / 255.0
        point = np.array(point, dtype=np.float32) / 255.0
        PF_point = np.array(PF_point, dtype=np.float32) / 255.0

        if len(mask.shape) > 2:
            mask = mask[:, :, 0]
            point = point[:, :, 0]
            PF_point = PF_point[:, :, 0]

        h, w = img.shape
        img = PadImg(img)
        mask = PadImg(mask)
        point_input = PadImg(point)
        PF_point = PadImg(PF_point)
        k, b = img.shape
        # **************************************************************************
        #                 Target Energy Initialization (TEI)
        # **************************************************************************
        if self.sigma > 0:
            mask_p = measure.label(point_input, connectivity=2)
            coord_m = measure.regionprops(mask_p)
            point_input_p = np.zeros(shape=(k, b), dtype=np.float32)
            for i in range(len(coord_m)):
                aa = np.array(list(coord_m[i].centroid))
                x = round(aa[0])
                y = round(aa[1])
                point_input_p[x, y] = 1
            point_input = Gaussian_dis(point_input_p, self.sigma)

            PF_point = PF_point[np.newaxis, :]
            PF_point = torch.from_numpy(np.ascontiguousarray(PF_point))

        elif self.sigma == 0:
            # get the central
            mask_p = measure.label(point_input, connectivity=2)
            coord_m = measure.regionprops(mask_p)
            point_input_p = np.zeros(shape=(k, b), dtype=np.float32)
            for i in range(len(coord_m)):
                aa = np.array(list(coord_m[i].centroid))
                x = round(aa[0])
                y = round(aa[1])
                point_input_p[x, y] = 1
            point_input_p = point_input_p[np.newaxis, :]  # 升维
            point_input = torch.from_numpy(np.ascontiguousarray(point_input_p))  # numpy 转tensor

            PF_point = PF_point[np.newaxis, :]
            PF_point = torch.from_numpy(np.ascontiguousarray(PF_point))
        else:
            point_input = point_input[np.newaxis, :]
            point_input = torch.from_numpy(np.ascontiguousarray(point_input))
            PF_point = PF_point[np.newaxis, :]
            PF_point = torch.from_numpy(np.ascontiguousarray(PF_point))

        img_patch, mask_patch = img[np.newaxis, :], mask[np.newaxis, :]

        # ************** increase dim, numpy->tensor ***************************
        img_patch = torch.from_numpy(np.ascontiguousarray(img_patch))
        mask_patch = torch.from_numpy(np.ascontiguousarray(mask_patch))
        if img_patch.size() != mask_patch.size():
            print('111')
        return img_patch, mask_patch, point_input, PF_point, [h, w], self.test_list[idx]

    def __len__(self):
        return len(self.test_list)