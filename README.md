### A Unified Stochastic Robust Gradient Reweighing Framework for Data Imbalance and Label Noise


To replicate the results for Data Imbalance:

CIFAR10
'''
python3 -W ignore train.py --dataset cifar10 --model resnet32 --epochs 200 --batch_size 100 --gpu 1 --loss_type robce --print_freq 100 --lamda 5 --init_lamda 200 --imb_factor 0.1 --seed 0 --CB_shots 160 --lr 0.1 --drogamma 0.7 --robAlpha 0.5 --imb_type exp --train_rule reweight --DP 0.2;
'''.


CIFAR100
'''
python3 -W ignore train.py --dataset cifar100 --model resnet32 --epochs 200 --batch_size 128 --gpu 3 --loss_type robce --print_freq 100 --lamda 3 --init_lamda 200 --imb_factor 0.01 --seed 0 --CB_shots 160 --lr 0.1 --drogamma 0.45 --robAlpha 0.3 --imb_type step --train_rule reweight;
'''


To replicate the results for Label Noise problem on Clothing1M:
'''
CUDA_VISIBLE_DEVICES=0 /home/qiuzh/.conda/envs/vlp/bin/python -W ignore train.py --lamda -5 --dataset_type clothing1M --batch_size 128 --epoch 10 --droGamma 0.5 --l2_reg 1e-3 --lr 0.005 --alpha 0.1 --nr 0.4 --robAlpha 0.5 --loss ROBSCE --class_tau 0 --version Q_ROBSCE0.4_cloth1M_-5_drogamma_0.5_lr_0.005_clt_0;
CUDA_VISIBLE_DEVICES=6 /home/qiuzh/.conda/envs/vlp/bin/python -W ignore train.py --lamda -5 --dataset_type clothing1M --batch_size 128 --epoch 10 --droGamma 0.5 --l2_reg 1e-3 --lr 0.005 --alpha 0.1 --nr 0.4 --robAlpha 0.5 --loss ROBCE --class_tau 0 --version Q_ROBCE0.4_cloth1M_-5_drogamma_0.5_lr_0.005_clt_0;
CUDA_VISIBLE_DEVICES=7 /home/qiuzh/.conda/envs/vlp/bin/python -W ignore train.py --lamda -5 --dataset_type clothing1M --batch_size 128 --epoch 10 --droGamma 0.5 --l2_reg 1e-3 --lr 0.005 --alpha 0.1 --nr 0.4 --robAlpha 0.5 --loss ROBTCE --class_tau 0 --version Q_ROBTCE0.4_cloth1M_-5_drogamma_0.5_lr_0.005_clt_0;
'''
