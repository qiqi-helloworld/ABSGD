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
from myutils import *
from sklearn.decomposition import PCA
import torch

#
# def my_warn():
#     pass
# warnings.warn = my_warn
#

class ROBSGD(nn.Module):
    '''
    Encoding ROBSGD algorithm with different loss functions.
    '''

    def __init__(self, args, loss_type, robAlpha = 0.5):
        '''
        :param threshold: margin for squred hinge loss
        '''
        super(ROBSGD, self).__init__()
        self.loss_type = loss_type
        self.u = 0
        self.u_1 = 0
        self.gamma = args.drogamma
        self.robAlpha = robAlpha
        self.criterion = CBCELoss(reduction='none')
        if 'ldam' in self.loss_type:
            # print(args.cls_num_list)
            self.criterion = LDAMLoss(cls_num_list=args.cls_num_list, max_m=0.5, s=30, reduction = 'none')
        elif 'focal' in self.loss_type:
            self.criterion = FocalLoss(gamma=args.gamma, reduction='none')

    def forward(self, output, target, cls_weights, myLambda):

        loss = self.criterion(output, target, cls_weights)
        # print("Before >>>:", loss.max(), ":<<<")
        # print("After >>>:", loss.max(), ":<<<")
        if myLambda >= 200: # reduces to CE
            p = 1/len(loss)
        else:
            # print(torch.mean(loss).item())
            # max_loss = max(loss)
            # print(max_loss.item())
            # if cls_weights is not None: # just for ldam right now
            #     loss = loss/torch.sum(cls_weights[target])

            # print(loss.max(), loss.min())
            expLoss = torch.exp(loss / myLambda)
            # u  = (1 - gamma) * u + gamma * alpha * g
            self.u = (1 - self.gamma) * self.u + self.gamma * (self.robAlpha * torch.mean(expLoss))
            drop = expLoss/(self.u * len(loss))
            drop.detach_()
            p = drop

            # if you want further boosting, you can try the following trick
            # print("qiqi:", len(p[p < 1 / loss.size(0)]), p.sum())
            # average_p = torch.ones_like(p)/loss.size(0)
            # p[p<1/loss.size(0)] = average_p[p<1/loss.size(0)]
            # p = p + 1 / loss.size(0)

        weighted_loss = torch.sum(p * loss)
        # print(weighted_loss.item())
        return weighted_loss

def get_train_loss(args, loss_type):
    if args.loss_type == 'ce':
        criterion = CBCELoss()
    elif args.loss_type == 'ldam':
        criterion = LDAMLoss(cls_num_list=args.cls_num_list, max_m=0.5, s=30)
    elif args.loss_type == 'focal':
        criterion = FocalLoss(gamma=1)
    elif 'rob' in args.loss_type:
        # print(' args.loss_type: ', args.loss_type, args.cls_num_list)
        criterion = ROBSGD(args, args.loss_type, robAlpha=args.robAlpha)
    elif 'neb' in args.loss_type:
        criterion = NEBLoss(args, args.loss_type, topK=args.topK, neb_tau= args.neb_tau, robAlpha=args.robAlpha)
    else:
        warnings.warn('Loss type is not listed')
        return

    return criterion

def focal_loss(input_values, alpha, gamma, reduction = 'mean'):
    """Computes the focal loss"""

    '''
    input_values = -\log(p_t)
    loss = - \alpha_t (1-\p_t)\log(p_t)
    '''
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


class NEBLoss(nn.Module):
    def __init__(self,args, loss_type, neb_tau = 1, topK = 5, robAlpha=1):
        super(NEBLoss, self).__init__()
        self.criterion = CBCELoss(reduction='none')
        self.robAlpha = robAlpha
        self.gamma = args.drogamma
        self.u = 0
        self.topK = topK
        self.neb_tau = neb_tau
        print(">>>>>>:", self.topK, ":<<<<<")
        self.cnt = 0
    def forward(self, output, target, cls_weights, myLambda, embed):

        loss = self.criterion(output, target, cls_weights)


        # print(weights_per_loss)
        # print(embed.sum(dim = 1, keepdim = True))
        # if we normalize, we ignore the magnititude of the mebedding, is it a good thing or bad thing?
        # print(loss.size(0))
        # print(each_sample_p.sum(dim = 1, keepdim = True),  weights_per_loss.max(), weights_per_loss.min())
        if myLambda >= 200:  # reduces to CE
            p = 1 / len(loss)
            embed_loss = loss
        else:
            embed = np.array(embed.detach().cpu())
            # TSNE:
            # embed_tsne = TSNE(n_components=2).fit_transform(embed)
            # PCA
            embed_tsne = PCA(n_components=2).fit_transform(embed)
            #
            embed_tsne = torch.tensor(embed_tsne).cuda()
            norm_embed = (1 + embed_tsne / embed_tsne.norm(dim=1, keepdim=True)) / 2  # batch size * feature dimension
            # print(embed_tsne.size())
            # embed = embed.detach()
            # norm_embed = embed/embed.norm(dim=1, keepdim=True)
            # print(">>>>>>> :", norm_embed, ": <<<<<<")
            sim_mat = norm_embed.matmul(norm_embed.T)  # batch size * batch size
            myTopK = self.topK

            top_sim_mat = torch.zeros_like(sim_mat)
            # print("Before: ", top_sim_mat)
            for i in range(top_sim_mat.size(0)):
                col_ind = sim_mat.topk(myTopK)[1][i]
                top_sim_mat[i][col_ind] = sim_mat[i][col_ind]

            topK_sim_mat_indices = top_sim_mat != 0
            exp_sim_mat = torch.exp(sim_mat / self.neb_tau) * topK_sim_mat_indices
            each_sample_p = exp_sim_mat / (
            exp_sim_mat.sum() / exp_sim_mat.size(0))  # batch size * batch size, each row summation eauals to 1
            weights_per_loss = each_sample_p.sum(0, keepdim=True)
            embed_loss = each_sample_p.matmul(loss)

            # print(torch.mean(loss).item())
            # max_loss = max(loss)
            # print(max_loss.item())
            # if cls_weights is not None: # just for ldam right now
            #     loss = loss/torch.sum(cls_weights[target])
            expLoss = torch.exp(embed_loss/myLambda)
            # u  = (1 - gamma) * u + gamma * alpha * g
            self.u = (1 - self.gamma) * self.u + self.gamma * (self.robAlpha * torch.mean(expLoss))
            drop = expLoss / (self.u * len(embed_loss))
            drop.detach_()
            p = drop

            # average_p = torch.ones_like(p) / loss.size(0)
            # print(len(p[p<1/loss.size(0)]))
            # p[p < 1 / loss.size(0)] = average_p[p < 1 / loss.size(0)]
            #
            # # p = p + 1/len(loss)
            # # print(torch.sum(p < 1 / loss.size(0)))

        # print(torch.max(embed.norm(dim = 1, keepdim = True)).item(), torch.min(sim_mat).item(), torch.max(sim_mat).item())
        # print(weights_per_loss)
        # print(loss)
        # print(embed_loss)
        # print(torch.sum(p*embed_loss).item(), torch.sum(p*loss).item())
        return torch.sum(p * embed_loss)
