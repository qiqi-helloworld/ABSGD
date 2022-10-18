__author__ = 'Qi'
# Created by on 10/29/20.
import warnings
from sklearn.exceptions import DataConversionWarning
warnings.filterwarnings(action='ignore')
from collections import Counter
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from utils import *
from sklearn.decomposition import PCA
import torch

class ABSGD(nn.Module):

    def __init__(self, args, loss_type, abAlpha = 0.5):
        super(ABSGD, self).__init__()
        self.loss_type = loss_type
        self.u = 0
        self.gamma = args.drogamma
        self.abAlpha = abAlpha
        self.criterion = CBCELoss(reduction='none')
        if 'ldam' in self.loss_type:
            # print(args.cls_num_list)
            self.criterion = LDAMLoss(cls_num_list=args.cls_num_list, max_m=0.5, s=30, reduction = 'none')
        elif 'focal' in self.loss_type:
            self.criterion = FocalLoss(gamma=args.gamma, reduction='none')

    def forward(self, output, target, cls_weights = None, myLambda = 200):

        loss = self.criterion(output, target, cls_weights)
        if myLambda >= 200: # reduces to CE
            p = torch.tensor(1/len(loss))
        else:
            expLoss = torch.exp(loss / myLambda)
            # u  = (1 - gamma) * u + gamma * alpha * g
            self.u = (1 - self.gamma) * self.u + self.gamma * (self.abAlpha * torch.mean(expLoss))
            drop = expLoss/(self.u * len(loss))
            drop.detach_()
            p = drop

        return torch.sum(p * loss)


def focal_loss(input_values, alpha, gamma, reduction = 'mean'):
    """Computes the focal loss"""

    p = torch.exp(-input_values)
    loss = alpha * (1 - p) ** gamma * input_values

    if reduction == 'none':
        return loss
    else:
        return loss.mean()

class FocalLoss(nn.Module):
    def __init__(self, alpha = 1, gamma=0, reduction = 'mean'):
        super(FocalLoss, self).__init__()
        assert gamma >= 0
        self.gamma = gamma
        self.reduction = reduction
        self.alpha = alpha

    def forward(self, input, target, weight=None):
        return focal_loss(F.cross_entropy(input, target, reduction='none', weight=weight), self.alpha, self.gamma, reduction = self.reduction)

class LDAMLoss(nn.Module):
    def __init__(self, cls_num_list, max_m=0.8, s=30, reduction='mean'):
        super(LDAMLoss, self).__init__()
        m_list = 1.0 / np.sqrt(np.sqrt(cls_num_list))  # 1/n_j^{1/4}
        m_list = m_list * (max_m / np.max(m_list)) # control the length of margin
        m_list = torch.cuda.FloatTensor(m_list)
        self.m_list = m_list
        assert s > 0
        self.s = s
        self.reduction = reduction

    def forward(self, output, target, weight):
        index = torch.zeros_like(output, dtype=torch.uint8)
        target = target.type(torch.cuda.LongTensor)
        index.scatter_(1, target.data.view(-1, 1), 1)
        index_float = index.type(torch.cuda.FloatTensor)

        batch_m = torch.matmul(self.m_list[None, :], index_float.transpose(0, 1))
        batch_m = batch_m.view((-1, 1))
        x_m = output - batch_m

        output = torch.where(index, x_m, output)

        return F.cross_entropy(self.s * output, target, weight=weight, reduction=self.reduction)

class CBCELoss(nn.Module):
    def __init__(self, reduction = 'mean'):
        super(CBCELoss, self).__init__()
        self.reduction = reduction
    def forward(self, out, target, weight = None):
        criterion = nn.CrossEntropyLoss(weight=weight, reduction=self.reduction)
        cbloss = criterion(out, target)
        return cbloss


def get_train_rule_hyperparameters(args, train_rule = None, epoch = 0):
    '''
    :param args: all the info needed for training strategy.
    :param train_rule: which training strategy we applied to address the data imbalance issue
    :param epoch: this epoch used to decide whether we apply the training rule for the current epoch.
    :return: the hyperparameters of the corresponding training rule.
    '''

    cls_weights = None
    train_sampler = None
    if train_rule == 'None':
        train_sampler = None
        cls_weights = None
    elif 'reweight' in train_rule:
        if args.train_defer:
            idx = epoch // args.CB_shots
            betas = [args.beta] * args.epochs
            betas[0] = 0
            effective_num = 1.0 - np.power(betas[idx], args.cls_num_list)
            cls_weights = (1.0 - betas[idx]) / np.array(effective_num)
            cls_weights = cls_weights / np.sum(cls_weights) * len(args.cls_num_list)
            cls_weights = torch.FloatTensor(cls_weights).cuda()
        else:
            beta = args.beta
            effective_num = 1.0 - np.power(beta, args.cls_num_list)
            cls_weights = (1.0 - beta) / np.array(effective_num)
            cls_weights = cls_weights / np.sum(cls_weights) * len(args.cls_num_list)
            cls_weights = torch.FloatTensor(cls_weights).cuda()

    return cls_weights, train_sampler
