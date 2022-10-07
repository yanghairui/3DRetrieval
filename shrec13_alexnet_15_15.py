import argparse
import os
import sys

import yaml
# import matplotlib.pyplot as plt
import numpy as np
import random
import torch.nn as nn
import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
# from torch.nn.utils import clip_grad_norm_
from torch.optim import lr_scheduler
from tensorboardX import SummaryWriter
from easydict import EasyDict

from dataset.shrec_2013 import shrec_view
from dataset.shrec_view import MakeDataLoader
from models.mvcnn import resnet18
from models.alexnet import alexnet
from models.norm import l2
from loss import TripletCenterLoss
# Triplet-Center Loss(TCL)
from loss import TCLNParamsLoss
from LR import CosineAnnealing
from utils import calc_acc


GPU_ID = '0'
os.environ['CUDA_VISIBLE_DEVICES'] = GPU_ID

parser = argparse.ArgumentParser(description='SHREC 2013 - Sketch based 3D Object Retrieval')
parser.add_argument('--config', default='experiments/shrec_2013/model/alexnet/config.yaml')
args = parser.parse_args()

with open(args.config) as rPtr:
    config = EasyDict(yaml.load(rPtr))
config.save_path = os.path.dirname(args.config)

torch.manual_seed(config.SEED)
torch.cuda.manual_seed(config.SEED)
np.random.seed(config.SEED)
random.seed(config.SEED)

writer = SummaryWriter(config.save_path + '/events')

train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
val_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
ViewTrainSet = shrec_view(path=config.PATH, phase='all', transform=train_transform)
ViewTrainLoader = MakeDataLoader(dataset=ViewTrainSet, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=10)

ViewValSet = shrec_view(path=config.PATH, phase='val', transform=val_transform)
ViewValLoader = MakeDataLoader(dataset=ViewValSet, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=10)

# assert config.MODEL_TYPE in (34, 50)
# if config.MODEL_TYPE == 34:
#     model = mvcnn.resnet34(
#         config.PRETRAINED,
#         num_classes=config.NUM_CLASSES,
#         view_ensemble=config.VIEW_ENSEMBLE,
#         num_views=config.NUM_VIEWS)
#     config.MODEL_FEA_DIM = 512
# elif config.MODEL_TYPE == 50:
#     model = mvcnn.resnet50(
#         config.PRETRAINED,
#         num_classes=config.NUM_CLASSES,
#         view_ensemble=config.VIEW_ENSEMBLE,
#         num_views=config.NUM_VIEWS)
#     config.MODEL_FEA_DIM = 2048
# else:
#     print('`MODEL_TYPE` parameter must be 34(ResNet34) or 50(ResNet50)')
#     sys.exit(-1)

print(config.VIEW_ENSEMBLE)
model = alexnet(pretrained=config.PRETRAINED,
                num_classes=config.NUM_CLASSES, view_ensemble=config.VIEW_ENSEMBLE, num_views=config.NUM_VIEWS)
# model = nn.DataParallel(model, device_ids=[0, 2, 3, 4])

model = model.cuda()
cudnn.benchmark = True

cross_entropy_loss = nn.CrossEntropyLoss(weight=None)
triplet_center_loss = TripletCenterLoss.TripletCenterCosineLoss(
    config=config,
    num_classes=config.NUM_CLASSES,
    fea_dim=config.MODEL_FEA_DIM,
    margin=config.MARGIN,
    l2Norm=l2.L2Norm())
# optimizer_cel = optim.SGD(model.parameters(), lr=LR_CNN,
#    momentum=MOMENTUM,
#    weight_decay=WEIGHT_DECAY)
optimizer_cel = None
# optimizer_ltr = optim.SGD(linear_trans.parameters(), lr=LR_LT,
#    momentum=MOMENTUM,
#    weight_decay=WEIGHT_DECAY)
if True:
    # lt_params = list(map(id, model.lt.parameters()))
    lt_params = list()
    lt_params.extend(list(map(id, model.classifier.parameters())))
    # lt_params.extend(list(map(id, model.view_select.parameters())))
    # lt_params.extend(list(map(id, model.view_select2.parameters())))
    backbone_params = filter(
        lambda params: id(params) not in lt_params,
        model.parameters())
    params = [
        {'params': backbone_params, 'lr': float(config.LR_CNN)},
        {'params': model.classifier.parameters(), 'lr': float(config.LR_LT)},
        # {'params': model.classify.parameters(), 'lr': float(config.LR_LT)},
        # {'params': model.view_select.parameters(), 'lr': float(config.LR_LT)},
        # {'params': model.view_select2.parameters(), 'lr': LR_LT}
    ]
    optimizer_cel = optim.SGD(
        params,
        momentum=config.MOMENTUM,
        weight_decay=float(config.WEIGHT_DECAY))
optimizer_tcl = optim.SGD(triplet_center_loss.parameters(),
    lr=config.LR_TCL_CENTERS)
# scheduler_cel = lr_scheduler.StepLR(optimizer_cel, step_size=STEP_SIZE,
#     gamma=GAMMA)
global scheduler_cos, scheduler_tcl
scheduler_cos = CosineAnnealing(optimizer_cel, config.MAX_EPOCH * len(ViewTrainLoader))
scheduler_tcl = CosineAnnealing(optimizer_tcl, config.MAX_EPOCH * len(ViewTrainLoader))
# scheduler_tcl = lr_scheduler.StepLR(optimizer_tcl, step_size=STEP_SIZE,
#    gamma=GAMMA)
# scheduler_ltr = lr_scheduler.StepLR(optimizer_ltr, step_size=STEP_SIZE,
#    gamma=GAMMA)

PCA = calc_acc.PerClassAccuracy(config.NUM_CLASSES)
l2_feats_norm = l2.L2Norm()
def train_one_epoch(epoch_idx):
    model.train()
    sum_tcl_loss = 0.0
    sum_loss = 0.0
    for batch_idx, sample in enumerate(ViewTrainLoader):
        images, labels = sample

        images = images.cuda()
        labels = labels.type(torch.long).cuda()

        feats, out = model(images, None)
        # Transformation
        # trans_feats = linear_trans(feats)
        # Normalization
        feats = l2_feats_norm(feats)
        pred = out.data.max(1)[1]
        PCA.update(labels.data.cpu().numpy(), pred.data.cpu().numpy())

        cel_loss = cross_entropy_loss(out, labels)
        tcl_loss = triplet_center_loss(feats, labels)
        loss = config.TCL_WEIGHT * tcl_loss + cel_loss

        # loss = cross_entropy_loss(out, labels) + \
        #     TCL_WEIGHT * triplet_center_loss(feas, labels)
        sum_loss += loss.item()
        sum_tcl_loss += tcl_loss.item()

        optimizer_cel.zero_grad()
        optimizer_tcl.zero_grad()
        # optimizer_ltr.zero_grad()
        loss.backward()
        for param in triplet_center_loss.parameters():
            param.grad.data *= (1. / config.TCL_WEIGHT)
        # for param in linear_trans.parameters():
        #    param.grad.data *= (1. / TCL_WEIGHT)
        # clip_grad_norm_(
        #      triplet_center_loss.parameters(), CLIP_GRAD)

        optimizer_cel.step()
        optimizer_tcl.step()
        # optimizer_ltr.step()

        scheduler_cos.step()
        scheduler_tcl.step()

        writer.add_scalar(
            'triplet-center loss[train]',
            tcl_loss.item(),
            epoch_idx * len(ViewTrainLoader) + batch_idx)
        writer.add_scalar(
            'cross_entropy loss[train]',
            loss.item(),
            epoch_idx * len(ViewTrainLoader) + batch_idx)
        print("[TRAIN]: {}/{}, [BATCH INDEX]: {}/{}, TRAIN LOSS:{}, cel: {}, tcl: {}".format(
            epoch_idx, config.MAX_EPOCH, batch_idx, len(ViewTrainLoader), loss.item(), cel_loss.item(), tcl_loss.item()))

    per_class_acc, avg_acc, mAP = PCA.calc()
    for class_idx in range(config.NUM_CLASSES):
        writer.add_scalar(
            'train_class' + str(class_idx) + '_accu',
            per_class_acc[class_idx], epoch_idx)
    writer.add_scalar('train_accu', avg_acc, epoch_idx)
    # writer.add_scalar('mAP', mAP, epoch_idx)
    writer.add_scalar(
        'train_loss',
        sum_loss / len(ViewTrainLoader), epoch_idx)
    writer.add_scalar(
        'triplet-center loss avg[train]',
        sum_tcl_loss / len(ViewTrainLoader), epoch_idx)
    PCA.reset()

def val_one_epoch(epoch_idx):
    model.eval()
    sum_tcl_loss = 0.0
    sum_loss = 0.0
    for batch_idx, sample in enumerate(ViewValLoader):
        images, labels = sample

        images = images.cuda()
        labels = labels.type(torch.long).cuda()

        with torch.no_grad():
            feats, out = model(images, None)
            # Transformation
            # trans_feats = linear_trans(feats)
            # Normalization
            feats = l2_feats_norm(feats)
        pred = out.data.max(1)[1]
        PCA.update(labels.data.cpu().numpy(), pred.data.cpu().numpy())

        with torch.no_grad():
            cel_loss = cross_entropy_loss(out, labels)
            tcl_loss = triplet_center_loss(feats, labels)
            loss = config.TCL_WEIGHT * tcl_loss + cel_loss
            # loss = cel_loss
            # loss = cross_entropy_loss(out, labels) + \
            #     TCL_WEIGHT * triplet_center_loss(feas, labels)
        sum_loss += loss.item()
        sum_tcl_loss += tcl_loss.item()

        writer.add_scalar(
            'triplet-center loss[val]',
            tcl_loss.item(),
            epoch_idx * len(ViewValLoader) + batch_idx)
        writer.add_scalar(
            'cross_entropy loss[val]',
            cel_loss.item(),
            epoch_idx * len(ViewValLoader) + batch_idx)
        print("[VAL]: {}/{}, [BATCH INDEX]: {}/{}".format(epoch_idx, config.MAX_EPOCH, batch_idx, len(ViewValLoader)))

    per_class_acc, avg_acc, mAP = PCA.calc()
    for class_idx in range(config.NUM_CLASSES):
        writer.add_scalar(
            'val_class' + str(class_idx) + '_accu',
            per_class_acc[class_idx], epoch_idx)
    writer.add_scalar('val_accu', avg_acc, epoch_idx)
    writer.add_scalar('mAP', mAP, epoch_idx)
    writer.add_scalar(
        'val_loss',
        sum_loss / len(ViewValLoader), epoch_idx)
    writer.add_scalar(
        'triplet-center loss avg[val]',
        sum_tcl_loss / len(ViewValLoader), epoch_idx)
    PCA.reset()

    # if not os.path.isdir(CHECK_POINTS):
    #     os.mkdir(CHECK_POINTS)
    # torch.save(
    #    triplet_center_loss.state_dict(),
    #    os.path.join(
    #        CHECK_POINTS,
    #        'center_' + str(epoch_idx) + '.pth'))

    return avg_acc, mAP

def save_checkpoints(accuracy, mAP, epoch_idx, surffix=''):
    infos = {
        "MODEL": model.state_dict(),
        "ACCURACY": accuracy,
        "mAP": mAP,
        "EPOCH_IDX": epoch_idx,
        "CENTERS": triplet_center_loss.state_dict(),
        # "LINEAR_TRANS": linear_trans.state_dict()
    }
    if not os.path.isdir(os.path.join(config.save_path, config.CHECK_POINTS)):
        os.mkdir(os.path.join(config.save_path, config.CHECK_POINTS))
    torch.save(
        infos, os.path.join(config.save_path, config.CHECK_POINTS, 'AlexNet' + str(config.MODEL_TYPE) + surffix + '.pth'))

BEST_ACCURACY = 0.0
BEST_MAP = 0.0
for epoch_idx in range(config.MAX_EPOCH):
    train_one_epoch(epoch_idx=epoch_idx)
    accuracy, mAP = val_one_epoch(epoch_idx=epoch_idx)

    # scheduler_cel.step()
    # scheduler_cos.step()
    # scheduler_tcl.step()
    # scheduler_ltr.step()

    if accuracy >= BEST_ACCURACY:
        BEST_ACCURACY = accuracy
        save_checkpoints(BEST_ACCURACY, mAP, epoch_idx)
    if mAP >= BEST_MAP:
        BEST_MAP = mAP
        save_checkpoints(accuracy, BEST_MAP, epoch_idx, 'mAP')
    if epoch_idx == 59:
        save_checkpoints(accuracy, mAP, epoch_idx, 'all')
    # save_checkpoints(accuracy, epoch_idx, str(epoch_idx))

writer.close()
