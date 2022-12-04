# Attentional-Biased Stochastic Gradient Descent[![pdf](https://img.shields.io/badge/Arxiv-pdf-orange.svg?style=flat)](https://arxiv.org/pdf/2012.06951.pdf)
This is the official implementation of Algorithm 1 the paper "[**Attentional-Biased Stochastic Gradient Descent**](https://arxiv.org/pdf/2012.06951.pdf)".

<img src="https://user-images.githubusercontent.com/17371111/196511607-ade8c8ee-d07d-4dc4-9939-6d467bb5049e.png" alt="drawing" width="400"/>

 Key parameters of ABSGD
```python
--mylambda (default 0.5) : $\lambda$
--abgamma (default 0.9) : $\gamma$, moving average hyper-parameter
```
News<img src="https://user-images.githubusercontent.com/17371111/196532894-41de92a5-8ccb-48ed-b477-aa435e155c1f.png" alt="drawing" width="20"/>
----------------------------------------------
With the assistant of ABSGD, we achieve ***1st in ResNet50*** (4th of 16 in total) in the [iWildCam](https://wilds.stanford.edu/leaderboard/) out of distribution changllenge, Oct 2022. The code repo is provided in the [wilds-competition](https://github.com/qiqi-helloworld/ABSGD/tree/main/wilds-competition).

The package has been released, to install:
```
pip3 install absgd
```
Training tutorial and examples:
Package
----------
```python
>>> from absgd.losses import ABLoss
>>> from absgd.optimizers import ABSGD, ABAdam
```
You can design your own loss. The following is a usecase, for more details pelease refer [ABSGD_tutorial.ipynb](https://github.com/qiqi-helloworld/ABSGD/ABSGD_tutorial.ipynb).
```python
>>> #import library
>>> from absgd.losses import ABLoss
>>> from absgd.optimizers import ABSGD, ABAdam
...
>>> #define loss
>>> mylambda = 0.5
# this can be easily combined with existing CBCE, LDAM loss, please refer our paper https://arxiv.org/pdf/2012.06951.pdf
>>> criterion =  nn.CrossEntropyLoss(reduction='none') 
>>> abloss = ABLoss(mylambda, criterion = criterion)
>>> optimizer = ABSGD()
...
>>> #training
>>> model.train()
>>> for epoch in range(epochs):
>>>     for i, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.cuda(), targets.cuda()
        outputs = model(inputs)
        losses = abloss(outputs, targets)
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
    # for two-stage $\lambda$ updates
    abloss.updateLambda()
```


Reminder
----------
If you want to download the code that reproducing the reported table results for the [Attentional Biased Stochastic Gradient Descent](https://arxiv.org/pdf/2012.06951.pdf), please go to the subdirectories and refer next section!



Reproduce results for the paper!
----------------------------------------------
In the paper,  we combine ABSGD with other SOTA losses such as CBCE, LDAM, focal.

```python
        self.criterion = CBCELoss(reduction='none')
        if 'ldam' in self.loss_type:
            self.criterion = LDAMLoss(cls_num_list=args.cls_num_list, max_m=0.5, s=30, reduction = 'none')
        elif 'focal' in self.loss_type:
            self.criterion = FocalLoss(gamma=args.gamma, reduction='none')
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
