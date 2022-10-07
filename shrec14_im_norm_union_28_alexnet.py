# import matplotlib.pyplot as plt
import argparse
import os

import yaml
import numpy as np
import random
import torch.nn as nn
import torch.optim as optim
import torch
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from torch.optim import lr_scheduler
from tensorboardX import SummaryWriter
from easydict import EasyDict

from models.alexnet import alexnet
from models.resnet import resnet50
from models.norm import l2
from dataset.shrec_2014 import shrec_image
from dataset.shrec_img import MakeDataLoader
# from loss import TCLNParamsLoss
from LR import CosineAnnealing
from loss import CenterNParamsLoss
from utils import calc_acc


GPU_ID = '0'
os.environ['CUDA_VISIBLE_DEVICES'] = GPU_ID

parser = argparse.ArgumentParser(description='SHREC 2014 - Sketch based 3D Object Retrieval')
parser.add_argument('--config', default='experiments/shrec_2014/image/alexnet/config.yaml')
parser.add_argument('--center_path', default='experiments/shrec_2014/model/alexnet/checkpoints/AlexNet50all.pth')
args = parser.parse_args()

with open(args.config) as rPtr:
    config = EasyDict(yaml.load(rPtr))
config.save_path = os.path.dirname(args.config)

torch.manual_seed(config.SEED)
torch.cuda.manual_seed(config.SEED)
np.random.seed(config.SEED)
random.seed(config.SEED)

writer = SummaryWriter(config.save_path + '/events')


transform_train = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    # transforms.RandomVerticalFlip(),
    # transforms.RandomRotation(20),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
transform_val = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

ImageTrainSet = shrec_image(path=config.PATH, phase='train', transform=transform_train)
print(len(ImageTrainSet))
ImageTrainLoader = MakeDataLoader(dataset=ImageTrainSet, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=0)
ImageValSet = shrec_image(path=config.PATH, phase='test', transform=transform_val)
print(len(ImageValSet))
ImageValLoader = MakeDataLoader(dataset=ImageValSet, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=0)

# assert config.MODEL_TYPE in (34, 50)
# if config.MODEL_TYPE == 34:
#     model = resnet.resnet34(config.PRETRAINED, num_classes=config.NUM_CLASSES)
#     FEAT_DIM = 512
# else:
#     model = resnet.resnet50(config.PRETRAINED, num_classes=config.NUM_CLASSES)
#     FEAT_DIM = 2048

model = alexnet(config.PRETRAINED, cls=True, num_classes=config.NUM_CLASSES)
# model = resnet50(config.PRETRAINED, num_classes=config.NUM_CLASSES)

model = model.cuda()
cudnn.benchmark = True

cross_entropy_loss = nn.CrossEntropyLoss()
tcl_wo_params_loss = CenterNParamsLoss.CenterWoParamsCosineLoss(
    center_path=args.center_path,
    l2Norm=l2.L2Norm())

lt_params = list(map(id, model.classifier.parameters()))
lt_params.extend(list(map(id, model.classify.parameters())))
backbone_params = filter(
    lambda params: id(params) not in lt_params,
    model.parameters())
params = [
    {'params': backbone_params, 'lr': float(config.LR_CNN)},
    {'params': model.classifier.parameters(), 'lr': float(config.LR_LT)},
    {'params': model.classify.parameters(), 'lr': float(config.LR_LT)}
]
optimizer = optim.SGD(
    params,
    momentum=config.MOMENTUM,
    weight_decay=float(config.WEIGHT_DECAY))
# scheduler = lr_scheduler.StepLR(
#    optimizer,
#    step_size=STEP_SIZE,
#    gamma=GAMMA)
global scheduler
scheduler = CosineAnnealing(optimizer, config.MAX_EPOCH * len(ImageTrainLoader))
# scheduler_lt = lr_scheduler.StepLR(
#     lt_optimizer,
#     step_size=STEP_SIZE,
#     gamma=GAMMA)

PCA = calc_acc.PerClassAccuracy(
    num_classes=config.NUM_CLASSES)
def train_one_epoch(epoch_idx):
    model.train()
    sum_loss = 0.0
    sum_tcl_loss = 0.0
    for batch_idx, item in enumerate(ImageTrainLoader):
        images, labels = item
        images = images.cuda()
        labels = labels.type(torch.long).cuda()

        feats, out = model(images)
        # if LINEAR_TRANS:
        #     # Linear transformation
        #     trans_feats = linear_trans(feats)
        #     # Normalization
        #     trans_feats = l2Norm(trans_feats)
        pred = out.data.max(1)[1]
        PCA.update(labels.data.cpu().numpy(), pred.data.cpu().numpy())

        cel_loss = cross_entropy_loss(out, labels)
        tcl_loss = tcl_wo_params_loss(feats, labels)
        loss = config.TCL_WEIGHT * tcl_loss + cel_loss
        # loss = cel_loss
        sum_loss += loss.item()
        sum_tcl_loss += tcl_loss.item()

        optimizer.zero_grad()
        # lt_optimizer.zero_grad()
        loss.backward()

        # for param in linear_trans.parameters():
        #     param.grad.data *= (1. / TCL_WEIGHT)

        optimizer.step()
        # lt_optimizer.step()

        scheduler.step()

        writer.add_scalar(
            'loss[train]',
            loss.item(),
            epoch_idx * len(ImageTrainLoader) + batch_idx)
        writer.add_scalar(
            'tcl_loss[train]',
            tcl_loss.item(),
            epoch_idx * len(ImageTrainLoader) + batch_idx)
        print("[TRAIN]: {}/{}, [BATCH INDEX]: {}/{}, TRAIN LOSS: {}, CENTER Loss: {}".format(
            epoch_idx, config.MAX_EPOCH, batch_idx, len(ImageTrainLoader), loss.item(), tcl_loss.item()))

    per_class_acc, avg_acc, _ = PCA.calc()
    for class_idx in range(config.NUM_CLASSES):
        writer.add_scalar(
            'train_class' + str(class_idx) + '_accu',
            per_class_acc[class_idx], epoch_idx)
    writer.add_scalar('train_accu', avg_acc, epoch_idx)
    writer.add_scalar(
        'train_loss',
        sum_loss / len(ImageTrainLoader), epoch_idx)
    writer.add_scalar(
        'train_tcl_avg_loss',
        sum_tcl_loss / len(ImageTrainLoader), epoch_idx)
    PCA.reset()

def val_one_epoch(epoch_idx):
    model.eval()
    sum_loss = 0.0
    sum_tcl_loss = 0.0
    for batch_idx, item in enumerate(ImageValLoader):
        images, labels = item
        images = images.cuda()
        labels = labels.type(torch.long).cuda()

        with torch.no_grad():
            feats, out = model(images)
            # if LINEAR_TRANS:
            #     # Linear transformation
            #     trans_feats = linear_trans(feats)
            #     # Normalization
            #     trans_feats = l2Norm(trans_feats)
        pred = out.data.max(1)[1]
        PCA.update(labels.data.cpu().numpy(), pred.data.cpu().numpy())

        cel_loss = cross_entropy_loss(out, labels)
        tcl_loss = tcl_wo_params_loss(feats, labels)
        loss = config.TCL_WEIGHT * tcl_loss + cel_loss
        # loss = cel_loss
        sum_loss += loss.item()
        sum_tcl_loss += tcl_loss.item()

        writer.add_scalar(
            'loss[val]',
            loss.item(),
            epoch_idx * len(ImageValLoader) + batch_idx)
        writer.add_scalar(
            'tcl_loss[val]',
            tcl_loss.item(),
            epoch_idx * len(ImageValLoader) + batch_idx)
        print("[VAL]: {}/{}, [BATCH INDEX]: {}/{}, TEST Loss: {}, CENTER Loss: {}".format(
            epoch_idx, config.MAX_EPOCH, batch_idx, len(ImageValLoader), loss.item(), tcl_loss.item()))

    per_class_acc, avg_acc, mAP = PCA.calc()
    for class_idx in range(config.NUM_CLASSES):
        writer.add_scalar(
            'val_class' + str(class_idx) + '_accu',
            per_class_acc[class_idx], epoch_idx)
    writer.add_scalar('val_accu', avg_acc, epoch_idx)
    writer.add_scalar('mAP', mAP, epoch_idx)
    writer.add_scalar(
        'val_loss',
        sum_loss / len(ImageValLoader), epoch_idx)
    writer.add_scalar(
        'val_tcl_avg_loss',
        sum_tcl_loss / len(ImageValLoader), epoch_idx)
    PCA.reset()

    return avg_acc, mAP

def save_checkpoints(accuracy, mAP, epoch_idx, surffix=''):
    infos = {
        "MODEL": model.state_dict(),
        "ACCURACY": accuracy,
        "mAP": mAP,
        "EPOCH_IDX": epoch_idx,
        # "LINEAR_TRANS": linear_trans.state_dict()
    }
    if not os.path.isdir(os.path.join(config.save_path, config.CHECK_POINTS)):
        os.mkdir(os.path.join(config.save_path, config.CHECK_POINTS))
    torch.save(
        infos,
        os.path.join(config.save_path, config.CHECK_POINTS, 'AlexNet' + str(config.MODEL_TYPE) + surffix + '_refine.pth'))


BEST_ACCURACY = 0.0
BEST_MAP = 0.0
for epoch_idx in range(config.MAX_EPOCH):
    train_one_epoch(epoch_idx=epoch_idx)
    accuracy, mAP = val_one_epoch(epoch_idx=epoch_idx)

    # scheduler.step()
    # scheduler_lt.step()

    if accuracy >= BEST_ACCURACY:
        BEST_ACCURACY = accuracy
        save_checkpoints(BEST_ACCURACY, mAP, epoch_idx)
    if mAP >= BEST_MAP:
        BEST_MAP = mAP
        save_checkpoints(accuracy, BEST_MAP, epoch_idx, 'mAP')
    # save_checkpoints(accuracy, epoch_idx, str(epoch_idx))

writer.close()
