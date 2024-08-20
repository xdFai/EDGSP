# -*- coding: utf-8 -*-
# @Author  : Shuai Yuan
# @File    : test_LG_SCTransNet_PdFa.py
# @Software: PyCharm
# coding=utf-8
from torch.autograd import Variable
import argparse
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataset01 import *
import time
from collections import OrderedDict
from model.EDGSP_SCTransNet import EDGSP_SCTransNet as EDGSP_SCTransNet
import numpy as np
import torch
import model.Config as config
from skimage import measure

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
parser = argparse.ArgumentParser(description="PyTorch BasicIRSTD test")
parser.add_argument("--PF_mode", type=str, default='centroid', help=" centroid, coarse ")
parser.add_argument("--test_mode", type=str, default='centroid', help=" centroid, coarse")
parser.add_argument("--sigma_input", type=int, default=4, help="If use Gaussian sigma")
parser.add_argument("--model_names", default=['EDGSP_SCTransNet_PdFa_Saliency'], type=list,
                    help="model_name: 'ACM', 'Ours01', 'DNANet', 'ISNet', 'ACMNet', 'Ours01', 'ISTDU-Net', 'U-Net', 'RISTDnet'")
parser.add_argument("--pth_dirs", default=['SIRST3/EDGSP_SCTransNet_cen_best.pth.tar'],
                    type=list)
parser.add_argument("--dataset_dir", default=r'./datasets', type=str, help="train_dataset_dir")
parser.add_argument("--dataset_names", default=['SIRST3'], type=list,
                    help="dataset_name: 'NUAA-SIRST', 'NUDT-SIRST', 'IRSTD-1K', 'SIRST3', 'NUDT-SIRST-Sea'")
parser.add_argument("--save_img", default=False, type=bool, help="save image of or not, we also choose to save the final result or the saliency map")
parser.add_argument("--img_norm_cfg", default=None, type=dict,
                    help="specific a img_norm_cfg, default=None (using img_norm_cfg values of each dataset)")
parser.add_argument("--save_img_dir", type=str, default=r'./Result/',
                    help="path of saved image")
parser.add_argument("--save_log", type=str, default=r'./log/', help="path of saved .pth")
parser.add_argument("--threshold", type=float, default=0.5)



global opt
opt = parser.parse_args()


class PD_FA_P():
    def __init__(self, ):
        super(PD_FA_P, self).__init__()
        self.image_area_total = []
        self.image_area_match = []
        self.dismatch_pixel = 0
        self.all_pixel = 0
        self.PD = 0
        self.FA = 0
        self.Dis = 0
        self.target = 0

    def update(self, preds, labels, img, size):
        predits = np.array((preds).cpu()).astype('int64')
        labelss = np.array((labels).cpu()).astype('int64')

        image = measure.label(predits, connectivity=2)
        coord_image = measure.regionprops(image)
        label = measure.label(labelss, connectivity=2)
        coord_label = measure.regionprops(label)

        self.target += len(coord_label)
        self.image_area_total = []
        self.image_area_match = []
        self.distance_match = []
        self.dismatch = []

        for K in range(len(coord_image)):
            area_image = np.array(coord_image[K].area)
            self.image_area_total.append(area_image)


        for i in range(len(coord_label)):
            centroid_label = np.array(list(coord_label[i].centroid))
            for m in range(len(coord_image)):
                centroid_image = np.array(list(coord_image[m].centroid))
                distance = np.linalg.norm(centroid_image - centroid_label)
                area_image = np.array(coord_image[m].area)
                if distance < 3:
                    self.distance_match.append(distance)
                    self.image_area_match.append(area_image)

                    del coord_image[m]
                    break
        # Pd  Fa
        if len(coord_image) != 0:
            print('Fa : {}'.format(img))

        if len(self.distance_match) != len(coord_label):
            print('Pd : {}'.format(img))

        self.dismatch = [x for x in self.image_area_total if x not in self.image_area_match]
        self.dismatch_pixel += np.sum(self.dismatch)

        self.all_pixel += size[0]*size[1]
        self.FA += len(coord_image)
        self.PD += len(self.distance_match)
        for j in range(len(self.distance_match)):
            self.Dis += self.distance_match[j]

    def get(self):
        Final_PD = self.PD / self.target
        Final_FA = self.FA / self.target
        Final_Dis = self.Dis / self.PD
        Final_FAP = self.dismatch_pixel / self.all_pixel
        return Final_PD, Final_FA, Final_Dis, float(Final_FAP.cpu().detach().numpy())

    def reset(self):
        self.FA = np.zeros([self.bins + 1])
        self.PD = np.zeros([self.bins + 1])


class mIoU():

    def __init__(self):
        super(mIoU, self).__init__()
        self.reset()

    def update(self, preds, labels):
        correct, labeled = batch_pix_accuracy(preds, labels)
        inter, union = batch_intersection_union(preds, labels)
        self.total_correct += correct
        self.total_label += labeled
        self.total_inter += inter
        self.total_union += union

    def get(self):
        pixAcc = 1.0 * self.total_correct / (np.spacing(1) + self.total_label)
        IoU = 1.0 * self.total_inter / (np.spacing(1) + self.total_union)
        mIoU = IoU.mean()
        return float(pixAcc), mIoU

    def reset(self):
        self.total_inter = 0
        self.total_union = 0
        self.total_correct = 0
        self.total_label = 0


def batch_pix_accuracy(output, target):
    if len(target.shape) == 3:
        target = np.expand_dims(target.float(), axis=1)
    elif len(target.shape) == 4:
        target = target.float()
    else:
        raise ValueError("Unknown target dimension")

    assert output.shape == target.shape, "Predict and Label Shape Don't Match"
    predict = (output > 0).float()
    pixel_labeled = (target > 0).float().sum()
    pixel_correct = (((predict == target).float()) * ((target > 0)).float()).sum()
    assert pixel_correct <= pixel_labeled, "Correct area should be smaller than Labeled"
    return pixel_correct, pixel_labeled


def batch_intersection_union(output, target):
    mini = 1
    maxi = 1
    nbins = 1
    predict = (output > 0).float()
    if len(target.shape) == 3:
        target = np.expand_dims(target.float(), axis=1)
    elif len(target.shape) == 4:
        target = target.float()
    else:
        raise ValueError("Unknown target dimension")
    intersection = predict * ((predict == target).float())

    area_inter, _ = np.histogram(intersection.cpu(), bins=nbins, range=(mini, maxi))
    area_pred, _ = np.histogram(predict.cpu(), bins=nbins, range=(mini, maxi))
    area_lab, _ = np.histogram(target.cpu(), bins=nbins, range=(mini, maxi))
    area_union = area_pred + area_lab - area_inter

    assert (area_inter <= area_union).all(), \
        "Error: Intersection area should be smaller than Union area"
    return area_inter, area_union


# **************************************************************
#                Bounding Box-based Matching (BBM)
# **************************************************************
def PD_FA_B(preds, labels,img):
    preds = preds[0, 0, :, :]
    labels = labels[0, 0, :, :]
    predits = np.array((preds).cpu()).astype('int64')
    labelss = np.array((labels).cpu()).astype('int64')

    image = measure.label(predits, connectivity=2)
    coord_image = measure.regionprops(image)
    label = measure.label(labelss, connectivity=2)
    coord_label = measure.regionprops(label)

    for i in range(len(coord_label)):
        centroid_label = np.array(list(coord_label[i].centroid)).astype('int64')
        for m in range(len(coord_image)):
            centroid_image = np.array(list(coord_image[m].bbox)).astype('int64')
            if centroid_image[0]-1 <= centroid_label[0] <= centroid_image[2] and centroid_image[1]-1 <= centroid_label[1] <= centroid_image[3]:
                # print('111')
                del coord_image[m]
                break


    for j in range(len(coord_image)):

        clean_np = np.array(list(coord_image[j].coords))
        for l, m in clean_np:
            predits[l][m] = 0

    predits = torch.from_numpy(predits)
    predits = predits[None, None, :]
    return predits


def test():
    test_set = TestSetLoader_Gauss_for_PdFa(opt.dataset_dir, opt.train_dataset_name, opt.test_dataset_name,
                                            opt.sigma_input, opt.test_mode, opt.PF_mode,
                                            img_norm_cfg=opt.img_norm_cfg)

    test_loader = DataLoader(dataset=test_set, num_workers=1, batch_size=1, shuffle=False)

    IOU = mIoU()
    eval_PD_FA = PD_FA_P()
    config_vit = config.get_SCTrans_config()
    net = EDGSP_SCTransNet(config_vit, n_channels=3, mode='test', deepsuper=True)
    state_dict = torch.load(opt.pth_dir, map_location='cpu')
    new_state_dict = OrderedDict()

    for k, v in state_dict['state_dict'].items():
        name = k[6:]
        new_state_dict[name] = v
    #  Here, set the strict as False, because the Sobel operator is written in the deepmodel.
    net.load_state_dict(new_state_dict, strict=False)
    net.eval()
    tbar = tqdm(test_loader)
    with torch.no_grad():
        for idx_iter, (img, gt_mask, point_input, PF_point, size, img_dir) in enumerate(tbar):
            img = Variable(img)

            point_input = Variable(point_input)
            PF_point = Variable(PF_point)
            img = torch.cat([img, point_input], 1)

            pred = net.forward(img)

            pred = pred[:, :, :size[0], :size[1]]
            gt_mask = gt_mask[:, :, :size[0], :size[1]]
            PF_point = PF_point[:, :, :size[0], :size[1]]

            # ----------------------------------------------------------
            #             Fa_branch     Bounding Box-based Matching (BBM)
            # ----------------------------------------------------------
            pred_thres = pred > opt.threshold
            predits = PD_FA_B(pred_thres, PF_point, img_dir)
            IOU.update(predits, gt_mask)
            eval_PD_FA.update(predits[0, 0, :, :], gt_mask[0, 0, :, :], img=img_dir,size=size)

            # IOU.update((pred > 0.5), gt_mask)
            # eval_PD_FA.update((pred[0, 0, :, :] > opt.threshold).cpu(), gt_mask[0, 0, :, :], img=img_dir)

            # Choose A or B: save img or saliency Map !!!
            if opt.save_img == True:
                # # A:  Save Image !!!
                predict = (predits[0, 0, :, :] > opt.threshold).float().cpu()
                img_save = transforms.ToPILImage()(predict)

                # # B:  Save Saliency Map !!!!
                # img_save = transforms.ToPILImage()((pred[0, 0, :, :]).cpu())

                if not os.path.exists(opt.save_img_dir + opt.test_dataset_name + '/' + opt.model_name):
                    os.makedirs(opt.save_img_dir + opt.test_dataset_name + '/' + opt.model_name)
                img_save.save(
                    opt.save_img_dir + opt.test_dataset_name + '/' + opt.model_name + '/' + img_dir[0] + '.png')

        pixAcc, mIOU = IOU.get()
        results2 = eval_PD_FA.get()
        print("mIOU: {}".format(mIOU))
        print("PD, FA, Distance:\t" + str(results2))
        opt.f.write("PD, FA, Distance:\t" + str(results2) + '\n')


if __name__ == '__main__':
    opt.f = open(opt.save_log + 'test_' + (time.ctime()).replace(' ', '_').replace(':', '_') + '.txt', 'w')
    if opt.pth_dirs == None:
        for i in range(len(opt.model_names)):
            opt.model_name = opt.model_names[i]
            print(opt.model_name)
            opt.f.write(opt.model_name + '_400.pth.tar' + '\n')
            for dataset_name in opt.dataset_names:
                opt.dataset_name = dataset_name
                opt.train_dataset_name = opt.dataset_name
                opt.test_dataset_name = opt.dataset_name
                print(dataset_name)
                opt.f.write(opt.dataset_name + '\n')
                opt.pth_dir = opt.save_log + opt.dataset_name + '/' + opt.model_name + '_400.pth.tar'
                test()
            print('\n')
            opt.f.write('\n')
        opt.f.close()
    else:
        for model_name in opt.model_names:
            for dataset_name in opt.dataset_names:
                for pth_dir in opt.pth_dirs:
                    opt.test_dataset_name = dataset_name
                    opt.model_name = model_name
                    opt.train_dataset_name = pth_dir.split('/')[0]
                    print(pth_dir)
                    opt.f.write(pth_dir)
                    print(opt.test_dataset_name)
                    opt.f.write(opt.test_dataset_name + '\n')
                    opt.pth_dir = opt.save_log + pth_dir
                    test()
                    print('\n')
                    opt.f.write('\n')
        opt.f.close()
