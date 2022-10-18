import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np

class SCELoss(torch.nn.Module):
    def __init__(self, alpha, beta, num_classes=10, reduction = 'mean'):
        super(SCELoss, self).__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.alpha = alpha
        self.beta = beta
        self.num_classes = num_classes
        self.cross_entropy = torch.nn.CrossEntropyLoss(reduction = reduction)
        self.reduction = reduction

    def forward(self, pred, labels):
        # CCE
        ce = self.cross_entropy(pred, labels)

        # RCE
        pred = F.softmax(pred, dim=1)
        pred = torch.clamp(pred, min=1e-7, max=1.0)
        # print(self.num_classes, labels)
        label_one_hot = torch.nn.functional.one_hot(labels, self.num_classes).float().to(self.device)
        label_one_hot = torch.clamp(label_one_hot, min=1e-4, max=1.0)
        rce = (-1*torch.sum(pred * torch.log(label_one_hot), dim=1))
        if self.reduction == 'mean':
            loss = self.alpha * ce + self.beta * rce.mean()
        else:
            loss = self.alpha * ce + self.beta * rce

        return loss


class ABLoss(torch.nn.Module):

    def __init__(self,  loss, alpha, beta, num_classes=10, abAlpha = 0.5, droGamma = 1, trainset_size = 50000):
        '''
        :param threshold: margin for squred hinge loss
        '''
        super(ABLoss, self).__init__()
        self.loss = loss
        self.trainset_size = trainset_size
        self.u = torch.tensor(0)
        if 'SCE' in self.loss:
            self.criterion = SCELoss(alpha, beta, num_classes=num_classes, reduction = 'none')
        elif 'TCE' in self.loss:
            self.criterion = TruncatedLoss(trainset_size = trainset_size)
        elif 'CE' in self.loss:
            self.criterion = nn.CrossEntropyLoss(reduction = 'none')
        # self.criterion =
        self.abAlpha = abAlpha
        self.droGamma =droGamma

    def forward(self, output, target,  indexes, myLambda, cls_weights = None):

        if 'TCE' in self.loss:
            loss = self.criterion(output, target, indexes)
        else:
            loss = self.criterion(output, target)

        expLoss = torch.exp(loss/myLambda)

        if myLambda >= 200: # reduces to CE
              p = 1/len(loss)
        else:
            # print(torch.mean(loss).item())
            self.u = (1 - self.droGamma) * self.u + self.droGamma * (self.abAlpha * torch.mean(expLoss))
            p = expLoss/(self.u*len(loss))
            p.detach_()
            p = p + 1/len(loss)
        return torch.sum(p * loss)


class ProbLossStable(torch.nn.Module):
    def __init__(self, reduction='none', eps=1e-5):
        super(ProbLossStable, self).__init__()
        self._name = "Prob Loss"
        self._eps = eps
        self._softmax = nn.Softmax(dim=-1)
        self._nllloss = nn.NLLLoss(reduction='none')

    def forward(self, outputs, labels):
        return self._nllloss(self._softmax(outputs), labels)



div = 'KL'
if div == 'KL':
    def activation(x):
        return -torch.mean(x)

    def conjugate(x):
        return -torch.mean(torch.exp(x - 1.))

elif div == 'Jeffrey':
    def activation(x):
        return -torch.mean(x)


    def conjugate(x):
        return -torch.mean(x + torch.mul(x, x) / 4. + torch.mul(torch.mul(x, x), x) / 16.)

elif div == 'Pearson':
    def activation(x):
        return -torch.mean(x)


    def conjugate(x):
        return -torch.mean(torch.mul(x, x) / 4. + x)

elif div == 'Neyman':
    def activation(x):
        return -torch.mean(1. - torch.exp(x))


    def conjugate(x):
        return -torch.mean(2. - 2. * torch.sqrt(1. - x))

elif div == 'Jenson-Shannon':
    def activation(x):
        return -torch.mean(- torch.log(1. + torch.exp(-x))) - torch.log(torch.tensor(2.))


    def conjugate(x):
        return -torch.mean(x + torch.log(1. + torch.exp(-x))) + torch.log(torch.tensor(2.))

elif div == 'Total-Variation':
    def activation(x):
        return -torch.mean(torch.tanh(x) / 2.)


    def conjugate(x):
        return -torch.mean(torch.tanh(x) / 2.)

else:
    raise NotImplementedError("[-] Not Implemented f-divergence %s" % div)




class f_divergence(torch.nn.Module):
    def __init__(self, num_classes):
        super(f_divergence, self).__init__()
        self.criterion_prob = ProbLossStable().cuda()
        self.num_classes = num_classes
    def forward(self, output, output1, target):
        target2 = torch.randint(0, self.num_classes, (target.shape)).cuda()
        loss_reg = activation(-self.criterion_prob(output, target.long()))
        loss_peer = conjugate(-self.criterion_prob(output1, target2.long()))
        loss = loss_reg - loss_peer
        return loss


class TruncatedLoss(nn.Module):
    def __init__(self, q=0.7, k=0.5, trainset_size=50000):
        super(TruncatedLoss, self).__init__()
        self.q = q
        self.k = k
        self.weight = torch.nn.Parameter(data=torch.ones(trainset_size, 1), requires_grad=False).cuda()

    def forward(self, logits, targets, indexes):
        p = F.softmax(logits, dim=1)
        # print(p.size(),  torch.unsqueeze(targets, 1))
        Yg = torch.gather(p, 1, torch.unsqueeze(targets, 1))
        loss = ((1 - (Yg ** self.q)) / self.q) * self.weight[indexes] - ((1 - (self.k ** self.q)) / self.q) * self.weight[indexes]
        #print(Yg.size(), p.size())

        return loss

    def update_weight(self, logits, targets, indexes):
        p = F.softmax(logits, dim=1)
        Yg = torch.gather(p, 1, torch.unsqueeze(targets, 1))
        Lq = ((1 - (Yg ** self.q)) / self.q)
        Lqk = np.repeat(((1 - (self.k ** self.q)) / self.q), targets.size(0))
        Lqk = torch.from_numpy(Lqk).type(torch.cuda.FloatTensor)
        Lqk = torch.unsqueeze(Lqk, 1)

        condition = torch.gt(Lqk, Lq)
        self.weight[indexes] = condition.type(torch.cuda.FloatTensor)
