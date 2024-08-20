# -*- coding: utf-8 -*-
# @Author  : Shuai Yuan
# @File    : train_LG_SCTransNet.py
# @Software: PyCharm
# coding=utf-8
import argparse
import time
import os
import cv2
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
from dataset01 import *
from metrics import *
from utils import *
import model.Config as config
from torch.utils.tensorboard import SummaryWriter
from model.EDGSP_SCTransNet import EDGSP_SCTransNet as EDGSP_SCTransNet

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

parser = argparse.ArgumentParser(description="PyTorch BasicIRSTD train")
parser.add_argument("--model_names", default=['EDGSP_SCTransNet'], type=list, help="'ACM', 'ALCNet'")
parser.add_argument("--dataset_names", default=['SIRST3'], type=list)  # SIRST3： NUAA NUDT-SIRST IRSTD-1K
parser.add_argument("--optimizer_name", default='Adam', type=str, help="optimizer name: AdamW, Adam, Adagrad, SGD")
#  If aug, please increase the epochs.
parser.add_argument("--epochs", default=800, type=int, help="numbers of epoch")
parser.add_argument("--begin_test", default=350, type=int)
parser.add_argument("--every_test", default=2, type=int)
parser.add_argument("--every_print", default=10, type=int)
parser.add_argument("--dataset_dir", default=r'./datasets')
parser.add_argument("--batchSize", type=int, default=16, help="Training batch sizse")
parser.add_argument("--sigma", type=int, default=4, help="If use Gaussian")
parser.add_argument("--aug", type=bool, default=False, help="If use data augmentation")
parser.add_argument("--train_mode", type=str, default='centroid', help=" centroid , U04, U04_Silence, coarse")
parser.add_argument("--test_mode", type=str, default='centroid', help=" centroid , U04,  U04_Silence, coarse")
# ******************* Others   *******************
parser.add_argument("--patchSize", type=int, default=256, help="Training patch size")
parser.add_argument("--save", default=r'./log', type=str, help="Save path of checkpoints")
parser.add_argument("--log_dir", type=str, default="./otherlogs/EDGSP_SCTransNet", help='path of log files')
parser.add_argument("--img_norm_cfg", default=None, type=dict)
parser.add_argument("--threads", type=int, default=0, help="Number of threads for data loader to use")
parser.add_argument("--threshold", type=float, default=0.5, help="Threshold for test")
parser.add_argument("--seed", type=int, default=42, help="Threshold for test")
parser.add_argument("--resume", default=False, type=list, help="Resume from exisiting checkpoints (default: None)")

global opt
opt = parser.parse_args()
seed_pytorch(opt.seed)
config_vit = config.get_SCTrans_config()
print("---------------------------------------------------------------")
print('batchSize: {0} -- begin_test: {1} -- every_print: {2} -- every_test: {3}'.format(opt.batchSize, opt.begin_test,
                                                                                        opt.every_print,
                                                                                        opt.every_test))


def train():
    train_set = TrainSetLoader_Gauss(dataset_dir=opt.dataset_dir, dataset_name=opt.dataset_name,
                                     patch_size=opt.patchSize, sigma=opt.sigma, train_mode=opt.train_mode, aug=opt.aug,
                                     img_norm_cfg=opt.img_norm_cfg)
    print('sigma: {} --aug: {}  --train_mode: {}  --test_mode: {} '.format(opt.sigma, opt.aug, opt.train_mode,
                                                                           opt.test_mode))
    train_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=True)

    test_set = TestSetLoader_Gauss(opt.dataset_dir, opt.dataset_name, opt.dataset_name, opt.sigma, opt.test_mode,
                                   img_norm_cfg=opt.img_norm_cfg)
    test_loader = DataLoader(dataset=test_set, num_workers=0, batch_size=1, shuffle=False)

    net = Net(model_name=opt.model_name, mode='train').cuda()
    # net = Net(model_name=opt.model_name, mode='train')
    net.apply(weights_init_kaiming)
    net.train()

    epoch_state = 0
    total_loss_list = []
    total_loss_epoch = []

    if not os.path.exists(opt.log_dir):
        os.makedirs(opt.log_dir)
    writer = SummaryWriter(opt.log_dir)

    if opt.optimizer_name == 'Adam':
        opt.optimizer_settings = {'lr': 0.001}
        opt.scheduler_name = 'CosineAnnealingLR'
        opt.scheduler_settings = {'epochs': opt.epochs, 'eta_min': 1e-5, 'last_epoch': -1}

    opt.nEpochs = opt.scheduler_settings['epochs']

    optimizer, scheduler = get_optimizer(net, opt.optimizer_name, opt.scheduler_name, opt.optimizer_settings,
                                         opt.scheduler_settings)

    # *******************************************************************************************************
    #                                             Train
    # *******************************************************************************************************
    for idx_epoch in range(epoch_state, opt.nEpochs):
        results1 = [0, 0]
        results2 = [0, 1, 1, 1]
        for idx_iter, (img, gt_mask, point_input) in enumerate(train_loader):
            img, gt_mask, point_input = Variable(img).cuda(), Variable(gt_mask).cuda(), Variable(point_input).cuda()
            # img, gt_mask, point_input = Variable(img), Variable(gt_mask), Variable(point_input)
            if img.shape[0] == 1:
                continue
            imgin = torch.cat([img, point_input], dim=1)
            preds = net.forward(imgin)
            loss = net.loss(preds, gt_mask)
            total_loss_epoch.append(loss.detach().cpu())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        scheduler.step()

        if (idx_epoch + 1) % opt.every_print == 0:  # tensorboard : write train loss
            total_loss_list.append(float(np.array(total_loss_epoch).mean()))
            print(time.ctime()[4:-5] + ' Epoch---%d, total_loss---%f, lr---%f,'
                  % (idx_epoch + 1, total_loss_list[-1], scheduler.get_last_lr()[0]))
            opt.f.write(time.ctime()[4:-5] + ' Epoch---%d, total_loss---%f,\n'
                        % (idx_epoch + 1, total_loss_list[-1]))
            total_loss_epoch = []
            writer.add_scalar('loss', total_loss_list[-1], idx_epoch + 1)
            writer.add_scalar('lr', scheduler.get_last_lr()[0], idx_epoch + 1)

        if idx_epoch == 0:
            best_mIOU = results1
            best_Pd_Fa = results2

        if (idx_epoch + 1) >= opt.begin_test and (
                idx_epoch + 1) % opt.every_test == 0:  # tensorboard: write test evaluate
            # *******************************************************************************************************
            #                                             Test
            # *******************************************************************************************************
            net.eval()
            with torch.no_grad():
                eval_mIoU = mIoU()
                eval_PD_FA = PD_FA()
                test_loss = []
                for idx_iter, (img, gt_mask, point_input, size, _) in enumerate(test_loader):
                    img = Variable(img).cuda()
                    # img = Variable(img)
                    point_input = Variable(point_input).cuda()
                    # point_input = Variable(point_input)
                    imgin = torch.cat([img, point_input], dim=1)
                    pred = net.forward(imgin)
                    if isinstance(pred, tuple):
                        pred = pred[-1]
                    elif isinstance(pred, list):
                        pred = pred[-1]
                    else:
                        pred = pred
                    pred = pred[:, :, :size[0], :size[1]]
                    gt_mask = gt_mask[:, :, :size[0], :size[1]]
                    # if pred.size() != gt_mask.size():
                    #     print('1111')
                    # loss = net.loss(pred, gt_mask.cuda())
                    loss = net.loss(pred, gt_mask)
                    test_loss.append(loss.detach().cpu())
                    eval_mIoU.update((pred > opt.threshold).cpu(), gt_mask.cpu())
                    eval_PD_FA.update((pred[0, 0, :, :] > opt.threshold).cpu(), gt_mask[0, 0, :, :], size)
                test_loss.append(float(np.array(test_loss).mean()))
                results1 = eval_mIoU.get()
                results2 = eval_PD_FA.get()
                writer.add_scalar('mIOU', results1[-1], idx_epoch + 1)
                writer.add_scalar('Pd', results2[0], idx_epoch + 1)
                writer.add_scalar('Fa', results2[1], idx_epoch + 1)
                writer.add_scalar('Fa_P', results2[3], idx_epoch + 1)
                writer.add_scalar('testloss', test_loss[-1], idx_epoch + 1)

            # IOU
            if results1[1] > best_mIOU[1] and results2[0] >= best_Pd_Fa[0]:
                best_mIOU = results1
                best_Pd_Fa[0] = results2[0]
                print('------save the best model epoch', opt.model_name, '_%d ------' % (idx_epoch + 1))
                opt.f.write("the best model epoch \t" + str(idx_epoch + 1) + '\n')
                print("mIoU:\t" + str(results1[1]))
                print("testloss:\t" + str(test_loss[-1]))
                print("PD, FA, Distance, FA_Pix :\t" + str(results2))
                opt.f.write("mIoU: " + str(results1[1]) + '\n')
                opt.f.write("PD, FA, Distance, FA_Pix :\t" + str(results2) + '\n')
                best_IOU = format(results1[1], '.4f')
                best_Pd = format(results2[0], '.4f')
                best_Fa = format(results2[1], '.6f')
                best_Dis = format(results2[2], '.4f')
                best_Fa_Pix = format(results2[3], '.6f')

                # save_pth = opt.save + '/' + opt.dataset_name + '/' + opt.model_name + '_' + str(
                #     idx_epoch + 1) + '_' + str(best_Pd) + "_" + str(best_Fa) + "_" + str(best_Dis) + "_" + str(best_Fa_Pix) + "_" + '.pth.tar'
                save_pth = opt.save + '/' + opt.dataset_name + '/' + opt.model_name + '_' + str(
                    idx_epoch + 1) + '_' + str(best_IOU) + '_' + str(best_Pd) + "_" + str(best_Fa) + "_" + str(
                    best_Fa_Pix) + "_" + '.pth.tar'
                save_checkpoint({
                    'epoch': idx_epoch + 1,
                    'state_dict': net.state_dict(),
                    'total_loss': total_loss_list,
                }, save_pth)


            # Fa
            elif results2[1] <= best_Pd_Fa[1] and results2[3] < best_Pd_Fa[3]:
                best_Pd_Fa[1] = results2[1]
                best_Pd_Fa[3] = results2[3]
                print('------save the best model epoch', opt.model_name, '_%d ------' % (idx_epoch + 1))
                opt.f.write("the best model epoch \t" + str(idx_epoch + 1) + '\n')
                print("mIoU:\t" + str(results1[1]))
                print("testloss:\t" + str(test_loss[-1]))
                print("PD, FA, Distance, FA_Pix :\t" + str(results2))
                opt.f.write("mIoU: " + str(results1[1]) + '\n')
                opt.f.write("PD, FA, Distance, FA_Pix :\t" + str(results2) + '\n')
                best_IOU = format(results1[1], '.4f')
                best_Pd = format(results2[0], '.4f')
                best_Fa = format(results2[1], '.6f')
                best_Dis = format(results2[2], '.4f')
                best_Fa_Pix = format(results2[3], '.6f')

                # save_pth = opt.save + '/' + opt.dataset_name + '/' + opt.model_name + '_' + str(
                #     idx_epoch + 1) + '_' + str(best_Pd) + "_" + str(best_Fa) + "_" + str(best_Dis) + "_" + str(best_Fa_Pix) + "_" + '.pth.tar'
                save_pth = opt.save + '/' + opt.dataset_name + '/' + opt.model_name + '_' + str(
                    idx_epoch + 1) + '_' + str(best_IOU) + '_' + str(best_Pd) + "_" + str(best_Fa) + "_" + str(
                    best_Fa_Pix) + "_" + '.pth.tar'
                save_checkpoint({
                    'epoch': idx_epoch + 1,
                    'state_dict': net.state_dict(),
                    'total_loss': total_loss_list,
                }, save_pth)


            elif results2[0] == 1:
                print('------save the best model epoch', opt.model_name, '_%d ------' % (idx_epoch + 1))
                opt.f.write("the best model epoch \t" + str(idx_epoch + 1) + '\n')
                print("mIoU:\t" + str(results1[1]))
                print("testloss:\t" + str(test_loss[-1]))
                print("PD, FA, Distance, FA_Pix :\t" + str(results2))
                opt.f.write("mIoU: " + str(results1[1]) + '\n')
                opt.f.write("PD, FA, Distance, FA_Pix :\t" + str(results2) + '\n')
                best_IOU = format(results1[1], '.4f')
                best_Pd = format(results2[0], '.4f')
                best_Fa = format(results2[1], '.6f')
                best_Dis = format(results2[2], '.4f')
                best_Fa_Pix = format(results2[3], '.6f')

                save_pth = opt.save + '/' + opt.dataset_name + '/' + opt.model_name + '_' + str(
                    idx_epoch + 1) + '_' + str(best_IOU) + '_' + str(best_Pd) + "_" + str(best_Fa) + "_" + str(
                    best_Fa_Pix) + "_" + '.pth.tar'
                save_checkpoint({
                    'epoch': idx_epoch + 1,
                    'state_dict': net.state_dict(),
                    'total_loss': total_loss_list,
                }, save_pth)


def save_checkpoint(state, save_path):
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
    torch.save(state, save_path)
    return save_path


class Net(nn.Module):
    def __init__(self, model_name, mode):
        super(Net, self).__init__()
        self.model_name = model_name
        # *******************************************************************************************************
        #                                             Loss
        # *******************************************************************************************************
        self.cal_loss = nn.BCELoss(size_average=True)
        if model_name == 'EDGSP_SCTransNet':
            input_channels = 3
            if mode == 'train':
                self.model = EDGSP_SCTransNet(config_vit, n_channels=input_channels, n_classes=1, mode='train',
                                              deepsuper=True)
            else:
                self.model = EDGSP_SCTransNet(config_vit, n_channels=input_channels, n_classes=1, mode='test',
                                              deepsuper=True)
            print('input channels: {}'.format(input_channels))
            print("---------------------------------------------------------------")

    def forward(self, img):
        return self.model(img)

    def loss(self, preds, gt_masks):
        if isinstance(preds, list):
            loss_total = 0
            for i in range(len(preds)):
                pred = preds[i]
                gt_mask = gt_masks[i]
                loss = self.cal_loss(pred, gt_mask)
                loss_total = loss_total + loss
            return loss_total / len(preds)

        elif isinstance(preds, tuple):
            a = []
            for i in range(len(preds)):
                pred = preds[i]
                loss = self.cal_loss(pred, gt_masks)
                a.append(loss)
            loss_total = a[0] + a[1] + a[2] + a[3] + a[4] + a[5]
            return loss_total

        else:
            loss = self.cal_loss(preds, gt_masks)
            return loss


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('BatchNorm') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


if __name__ == '__main__':
    for dataset_name in opt.dataset_names:
        opt.dataset_name = dataset_name
        for model_name in opt.model_names:
            opt.model_name = model_name
            if not os.path.exists(opt.save):
                os.makedirs(opt.save)
            opt.f = open(opt.save + '/' + opt.dataset_name + '_' + opt.model_name + '_' + (time.ctime()).replace(' ',
                                                                                                                 '_').replace(
                ':', '_') + '.txt', 'w')
            print(opt.dataset_name + '\t' + opt.model_name)
            train()
            print('\n')
            opt.f.close()
