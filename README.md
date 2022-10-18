# A Unified Stochastic Robust Gradient Reweighting Framework for Data Imbalance and Label Noise [![pdf](https://img.shields.io/badge/Arxiv-pdf-orange.svg?style=flat)](https://arxiv.org/pdf/2012.06951.pdf)
This is the official implementation of Algorithm 1 the paper "**A Unified Stochastic Robust Gradient Reweighting Framework for Data Imbalance and Label Noise**".

<img src="https://user-images.githubusercontent.com/17371111/196511607-ade8c8ee-d07d-4dc4-9939-6d467bb5049e.png" alt="drawing" width="400"/>



Implementation Details.
----------------------------------------------
During the implementation, we encode the robust weight $\tilde{p}_i$ in Step 6 in Algorithm 1 into the loss function and then by optimizing the robust loss using SGD, we have ABSGD. Implemented in this way, with simple modifications, ABSGD can easily optimize other SOTA losses such as CBCE, LDAM, focal. Key parameters of ABSGD
```python
--myLambda : $\lambda$
--abAlpha : a hyperparamter between $(0,1]$ to handle the numerical issue may appear the second stage.
--drogamma : $\gamma$, moving average hyper-parameter

```


```python
class ABSGD(nn.Module):
    def __init__(self, args, loss_type, abAlpha = 1):
        super(ABSGD, self).__init__()
        self.loss_type = loss_type
        self.u = 0
        self.gamma = args.drogamma
        self.robAlpha = robAlpha
        self.criterion = CBCELoss(reduction='none')
        if 'ldam' in self.loss_type:
            self.criterion = LDAMLoss(cls_num_list=args.cls_num_list, max_m=0.5, s=30, reduction = 'none')
        elif 'focal' in self.loss_type:
            self.criterion = FocalLoss(gamma=args.gamma, reduction='none')

    def forward(self, output, target, cls_weights, myLambda):

        loss = self.criterion(output, target, cls_weights)
        if myLambda >= 200: # reduces to CE
            p = 1/len(loss)
        else:
            expLoss = torch.exp(loss / myLambda)
            self.u = (1 - self.gamma) * self.u + self.gamma * (self.abAlpha * torch.mean(expLoss))
            drop = expLoss/(self.u * len(loss))
            drop.detach_()
            p = drop

        abloss = torch.sum(p * loss)
        return abloss
 ```

Running Examples for Data Imabalace Settings
----------------------------------------------
CIFAR10 with exponential imbalance ratio $\rho = 0.1$
```
python3 -W ignore train.py --dataset cifar10 --model resnet32 --epochs 200 --batch_size 100 --gpu 1 --loss_type abce --print_freq 100 --lamda 5 --init_lamda 200 --imb_factor 0.1 --seed 0 --CB_shots 160 --lr 0.1 --drogamma 0.7 --abAlpha 0.5 --imb_type exp --train_rule reweight --DP 0.2;
```


CIFAR100 with step imbalance ratio $\rho = 0.01$
```
python3 -W ignore train.py --dataset cifar100 --model resnet32 --epochs 200 --batch_size 128 --gpu 3 --loss_type abce --print_freq 100 --lamda 3 --init_lamda 200 --imb_factor 0.01 --seed 0 --CB_shots 160 --lr 0.1 --drogamma 0.45 --abAlpha 0.3 --imb_type step --train_rule reweight;
```


Running Examples for Noisy Label Settings
----------------------------------------------
```
CUDA_VISIBLE_DEVICES=0 python3 -W ignore train.py --lamda -5 --dataset_type clothing1M --batch_size 128 --epoch 10 --droGamma 0.5 --l2_reg 1e-3 --lr 0.005 --alpha 0.1 --nr 0.4 --abAlpha 0.5 --loss ABSCE --class_tau 0 --version Q_ABSCE0.4_cloth1M_-5_drogamma_0.5_lr_0.005_clt_0;
CUDA_VISIBLE_DEVICES=6 python3 -W ignore train.py --lamda -5 --dataset_type clothing1M --batch_size 128 --epoch 10 --droGamma 0.5 --l2_reg 1e-3 --lr 0.005 --alpha 0.1 --nr 0.4 --abAlpha 0.5 --loss ABCE --class_tau 0 --version Q_ABCE0.4_cloth1M_-5_drogamma_0.5_lr_0.005_clt_0;
CUDA_VISIBLE_DEVICES=7 python3 -W ignore train.py --lamda -5 --dataset_type clothing1M --batch_size 128 --epoch 10 --droGamma 0.5 --l2_reg 1e-3 --lr 0.005 --alpha 0.1 --nr 0.4 --abAlpha 0.5 --loss ABTCE --class_tau 0 --version Q_ABTCE0.4_cloth1M_-5_drogamma_0.5_lr_0.005_clt_0;
```
