
# CUDA_VISIBLE_DEVICES=0 python3 -W ignore train.py --lamda -3 --dataset_type cifar10 --lr 0.1  --l2_reg 5e-3 --nr 0.2 --droGamma 0.9 --loss SCE --alpha 0.1 --beta 0.1 --lamda_shots 80 --version SCE0.2_CIFAR10_asym_-3_0.9_lr_0.1 --asym True &
# CUDA_VISIBLE_DEVICES=1 python3 -W ignore train.py --lamda -3 --dataset_type cifar10 --lr 0.1  --l2_reg 5e-3 --nr 0.3 --droGamma 0.9 --loss SCE --alpha 0.1 --beta 0.1 --lamda_shots 80 --version SCE0.3_CIFAR10_asym_-3_0.9_lr_0.1 --asym True &
# CUDA_VISIBLE_DEVICES=2 python3 -W ignore train.py --lamda -3 --dataset_type cifar10 --lr 0.1  --l2_reg 5e-3 --nr 0.4 --droGamma 0.9 --loss SCE --alpha 0.1 --beta 0.1 --lamda_shots 80 --version SCE0.4_CIFAR10_asym_-3_0.9_lr_0.1 --asym True &

# CUDA_VISIBLE_DEVICES=2 python3 -W ignore train.py --lamda -3 --dataset_type cifar100 --lr 0.1  --l2_reg 5e-3 --nr 0.2 --droGamma 0.9 --loss ABSCE --alpha 0.5 --beta 0.1 --lamda_shots 80 --version 0616_ABSCE0.2_CIFAR100_asym_-3_0.9_lr_0.1 --asym True --data_path /home/qqi7/ABSGD/data;
# CUDA_VISIBLE_DEVICES=1 python3 -W ignore train.py --lamda -3 --dataset_type cifar100 --lr 0.1  --l2_reg 5e-3 --nr 0.3 --droGamma 0.9 --loss ABSCE --alpha 0.1 --lamda_shots 80 --version 0616_ABSCE0.3_CIFAR100_asym_-3_0.9_lr_0.1 --asym True --data_path /home/qqi7/ABSGD/data;
# CUDA_VISIBLE_DEVICES=3 python3 -W ignore train.py --lamda -3 --dataset_type cifar100 --lr 0.1  --l2_reg 5e-3 --nr 0.4 --droGamma 0.9 --loss ABSCE --alpha 0.1 --lamda_shots 80 --version 0616_ABSCE0.4_CIFAR100_asym_-3_0.9_lr_0.1 --asym True --data_path /home/qqi7/ABSGD/data;





# final version --------------------

CUDA_VISIBLE_DEVICES=0 /home/qiuzh/.conda/envs/vlp/bin/python -W ignore train.py --lamda -3 --dataset_type cifar100 --lr 0.1  --l2_reg 5e-3 --nr 0.2 --droGamma 0.9 --loss SCE --alpha 0.7 --beta 0.1 --lamda_shots 80 --version 0619_SCE0.2_CIFAR100_asym_0.9_lr_0.1 --asym True &
CUDA_VISIBLE_DEVICES=1 /home/qiuzh/.conda/envs/vlp/bin/python -W ignore train.py --lamda -3 --dataset_type cifar100 --lr 0.1  --l2_reg 5e-3 --nr 0.3 --droGamma 0.9 --loss SCE --alpha 0.7 --beta 0.1 --lamda_shots 80 --version 0619_SCE0.3_CIFAR100_asym_0.9_lr_0.1 --asym True &
CUDA_VISIBLE_DEVICES=2 /home/qiuzh/.conda/envs/vlp/bin/python -W ignore train.py --lamda -3 --dataset_type cifar100 --lr 0.1  --l2_reg 5e-3 --nr 0.4 --droGamma 0.9 --loss SCE --alpha 0.7 --beta 0.1 --lamda_shots 80 --version 0616_SCE0.4_CIFAR100_asym_0.9_lr_0.1 --asym True &
CUDA_VISIBLE_DEVICES=2 /home/qiuzh/.conda/envs/vlp/bin/python -W ignore train.py --lamda -3 --dataset_type cifar100 --lr 0.1  --l2_reg 5e-3 --nr 0.4 --droGamma 0.9 --loss ABSCE --alpha 0.7 --beta 0.1 --lamda_shots 80 --version 0622_ABSCE0.4_CIFAR100_asym_0.9_lr_0.1 --asym True;
CUDA_VISIBLE_DEVICES=4 /home/qiuzh/.conda/envs/vlp/bin/python -W ignore train.py --lamda -3 --abAlpha 0.5 --dataset_type cifar100 --lr 0.1  --l2_reg 5e-3 --nr 0.2 --droGamma 0.9 --loss ABSCE --alpha 0.4 --beta 0.1 --lamda_shots 80 --version 0622_ABSCE0.2_CIFAR100_asym_0.9_lr_0.1 --asym True &
CUDA_VISIBLE_DEVICES=5 /home/qiuzh/.conda/envs/vlp/bin/python -W ignore train.py --lamda -3 --abAlpha 0.5 --dataset_type cifar100 --lr 0.1  --l2_reg 5e-3 --nr 0.3 --droGamma 0.9 --loss ABSCE --alpha 0.4 --beta 0.1 --lamda_shots 80 --version 0622_ABSCE0.3_CIFAR100_asym_0.9_lr_0.1 --asym True &
CUDA_VISIBLE_DEVICES=6 /home/qiuzh/.conda/envs/vlp/bin/python -W ignore train.py --lamda -3 --abAlpha 0.5 --dataset_type cifar100 --lr 0.1  --l2_reg 5e-3 --nr 0.4 --droGamma 0.9 --loss ABSCE --alpha 0.4 --beta 0.1 --lamda_shots 80 --version 0622_ABSCE0.4_CIFAR100_asym_0.9_lr_0.1 --asym True &


#cifar10 symmetric
#ABSCE
CUDA_VISIBLE_DEVICES=1 python3 -W ignore train.py --lamda -0.5 --dataset_type cifar10 --lr 0.01 --alpha 0.1 --beta 2 --l2_reg 5e-3 --nr 0.2 --droGamma 0.9 --loss ABSCE --lamda_shots 80 --version 0629_ABSCE0.2_CIFAR10_0.9_lr_0.01_tau_-0.5 --data_path /home/qqi7/ABSGD/data;
CUDA_VISIBLE_DEVICES=3 python3 -W ignore train.py --lamda -0.5 --dataset_type cifar10 --lr 0.01 --alpha 0.1 --beta 2 --l2_reg 5e-3 --nr 0.4 --droGamma 0.9 --loss ABSCE --lamda_shots 80 --version 0629_ABSCE0.4_CIFAR10_0.9_lr_0.01_tau_-0.5 --data_path /home/qqi7/ABSGD/data;


#SCE
CUDA_VISIBLE_DEVICES=3 python3 -W ignore train.py --lamda -1 --dataset_type cifar10 --lr 0.01 --alpha 0.2 --beta 2 --l2_reg 5e-3 --nr 0.2 --droGamma 0.9 --loss SCE --lamda_shots 80 --version 0628_SCE0.2_CIFAR10_0.9_lr_0.01 --data_path /home/qqi7/ABSGD/data;
CUDA_VISIBLE_DEVICES=0 python3 -W ignore train.py --lamda -1 --dataset_type cifar10 --lr 0.01 --alpha 0.2 --beta 2 --l2_reg 5e-3 --nr 0.4 --droGamma 0.9 --loss SCE --lamda_shots 80 --version 0629_SCE0.4_CIFAR10_0.9_lr_0.01 --data_path /home/qqi7/ABSGD/data;


#CE
CUDA_VISIBLE_DEVICES=3 python3 -W ignore train.py --lamda -1 --dataset_type cifar10 --lr 0.01 --alpha 0.2 --beta 2 --l2_reg 5e-3 --nr 0.2 --droGamma 0.9 --loss CE --lamda_shots 80 --version 0703_CE0.2_CIFAR10_0.9_lr_0.01 --data_path /home/qqi7/ABSGD/data;
CUDA_VISIBLE_DEVICES=0 python3 -W ignore train.py --lamda -1 --dataset_type cifar10 --lr 0.01 --alpha 0.2 --beta 2 --l2_reg 5e-3 --nr 0.4 --droGamma 0.9 --loss CE --lamda_shots 80 --version 0703_CE0.4_CIFAR10_0.9_lr_0.01 --data_path /home/qqi7/ABSGD/data;

#TCE
CUDA_VISIBLE_DEVICES=0 python3 -W ignore train.py --lamda -3 --dataset_type cifar10 --lr 0.01 --alpha 0.2 --beta 2 --l2_reg 1e-4 --nr 0.2 --droGamma 0.9 --loss TCE --lamda_shots 80 --version 0706_TCE0.2_CIFAR10_0.9_lr_0.01 --data_path /home/qqi7/ABSGD/data;
CUDA_VISIBLE_DEVICES=0 python3 -W ignore train.py --lamda -3 --dataset_type cifar10 --lr 0.01 --alpha 0.2 --beta 2 --l2_reg 1e-4 --nr 0.4 --droGamma 0.9 --loss TCE --lamda_shots 80 --version 0706_TCE0.4_CIFAR10_0.9_lr_0.01 --data_path /home/qqi7/ABSGD/data;



#cifar10 asymmetric
#ABSCE
# nr 0.4 82.61
CUDA_VISIBLE_DEVICES=0 python3 -W ignore train.py --lamda -2 --dataset_type cifar10 --lr 0.001 --alpha 6 --beta 2 --l2_reg 5e-3 --nr 0.4 --droGamma 0.9 --loss ABSCE --lamda_shots 80 --version 0705_Asym_ABSCE0.4_CIFAR10_0.9_lr_0.001_tau_-0.5 --asym True --data_path /home/qqi7/ABSGD/data --class_tau 1;
CUDA_VISIBLE_DEVICES=0 python3 -W ignore train.py --lamda -2 --dataset_type cifar10 --lr 0.001 --alpha 6 --beta 2 --l2_reg 5e-3 --nr 0.2 --droGamma 0.9 --loss ABSCE --lamda_shots 80 --version 0705_Asym_ABSCE0.4_CIFAR10_0.9_lr_0.001_tau_-0.5 --asym True --data_path /home/qqi7/ABSGD/data --class_tau 1;

#SCE
CUDA_VISIBLE_DEVICES=0 python3 -W ignore train.py --lamda -1 --dataset_type cifar10 --lr 0.001 --alpha 6 --beta 2 --l2_reg 5e-3 --nr 0.4 --droGamma 0.9 --loss SCE --lamda_shots 40 --version 0703_Asym_SCE0.4_CIFAR10_0.9_lr_0.001 --asym True --data_path /home/qqi7/ABSGD/data;
CUDA_VISIBLE_DEVICES=1 python3 -W ignore train.py --lamda -1 --dataset_type cifar10 --lr 0.01 --alpha 0.2 --beta 2 --l2_reg 5e-3 --nr 0.2 --droGamma 0.9 --loss SCE --lamda_shots 40 --version 0703_Asym_SCE0.2_CIFAR10_0.9_lr_0.01 --asym True --data_path /home/qqi7/ABSGD/data;

#CE
CUDA_VISIBLE_DEVICES=0 python3 -W ignore train.py --lamda -1 --dataset_type cifar10 --lr 0.001 --alpha 0.1 --beta 2 --l2_reg 5e-3 --nr 0.4 --droGamma 0.9 --loss CE --lamda_shots 80 --version 0703_Asym_CE0.4_CIFAR10_0.9_lr_0.001 --asym True --data_path /home/qqi7/ABSGD/data;
CUDA_VISIBLE_DEVICES=3 python3 -W ignore train.py --lamda -1 --dataset_type cifar10 --lr 0.001 --alpha 0.1 --beta 2 --l2_reg 5e-3 --nr 0.2 --droGamma 0.9 --loss CE --lamda_shots 80 --version 0703_Asym_CE0.2_CIFAR10_0.9_lr_0.001 --asym True --data_path /home/qqi7/ABSGD/data;

#TCE
CUDA_VISIBLE_DEVICES=1 python3 -W ignore train.py --lamda -3 --dataset_type cifar10 --lr 0.01 --alpha 0.2 --beta 2 --l2_reg 1e-4 --nr 0.2 --droGamma 0.9 --loss TCE --lamda_shots 80 --version 0706_TCE0.2_CIFAR10_0.9_lr_0.01 --asym True  --data_path /home/qqi7/ABSGD/data;
CUDA_VISIBLE_DEVICES=2 python3 -W ignore train.py --lamda -3 --dataset_type cifar10 --lr 0.01 --alpha 0.2 --beta 2 --l2_reg 1e-4 --nr 0.4 --droGamma 0.9 --loss TCE --lamda_shots 80 --version 0706_TCE0.4_CIFAR10_0.9_lr_0.01 --asym True --data_path /home/qqi7/ABSGD/data;


#cifar100 symmetric
# CE nr 0.2 68.21 | nr 0.4: 56.29
# CE nr 0.2 72.13 | nr 0.4: 67.02 (class_tau = 1)
CUDA_VISIBLE_DEVICES=3 python3 -W ignore train.py --lamda -1 --dataset_type cifar100 --lr 0.01 --alpha 0.1 --beta 2 --l2_reg 1e-2 --nr 0.2 --droGamma 0.9 --loss CE --lamda_shots 80 --version 0706_CE0.2_CIFAR100_0.9_lr_0.01 --data_path /home/qqi7/ABSGD/data --class_tau 1;
CUDA_VISIBLE_DEVICES=1 python3 -W ignore train.py --lamda -1 --dataset_type cifar100 --lr 0.01 --alpha 0.1 --beta 2 --l2_reg 1e-2 --nr 0.4 --droGamma 0.9 --loss CE --lamda_shots 80 --version 0706_CE0.4_CIFAR100_0.9_lr_0.01 --data_path /home/qqi7/ABSGD/data --class_tau 1;


#SCE
# SCE nr 0.2: 68.28 | nr 0.4 58.82  (lr 0.01, 6, 0.5)
CUDA_VISIBLE_DEVICES=2 python3 -W ignore train.py --lamda -2 --dataset_type cifar100 --lr 0.01 --alpha 6 --beta 0.5 --l2_reg 1e-2 --nr 0.2 --droGamma 0.9 --loss SCE --lamda_shots 80 --version 0706_SCE0.2_CIFAR100_0.9_lr_0.01 --data_path /home/qqi7/ABSGD/data --class_tau 1 &
CUDA_VISIBLE_DEVICES=3 python3 -W ignore train.py --lamda -1 --dataset_type cifar100 --lr 0.01 --alpha 6 --beta 0.5 --l2_reg 1e-2 --nr 0.4 --droGamma 0.9 --loss SCE --lamda_shots 80 --version 0706_SCE0.4_CIFAR100_0.9_lr_0.01 --data_path /home/qqi7/ABSGD/data --class_tau 1;


# ABSCE
# ABCE nr 0.2:  0.7186, 0.7173, 72.13 | nr 0.4  66.36  (lr 0.01, 6, 0.5  class_tau = 1)
# ABCE nr 0.2:  0.72.21 | nr 0.4  67.59  (lr 0.01, 1, 0,  class_tau = 1)
# ABCE nr 0.2:  0.72.21 | nr 0.4  67.59  (lr 0.01, 2, 0,  class_tau = 1)

CUDA_VISIBLE_DEVICES=1 python3 -W ignore train.py --lamda -5 --dataset_type cifar100 --lr 0.01 --alpha 2 --beta 0 --l2_reg 1e-2 --nr 0.2 --droGamma 0.5 --loss ABSCE --lamda_shots 80 --version 0706_ABSCE0.2_CIFAR100_0.9_lr_0.01 --data_path /home/qqi7/ABSGD/data --class_tau 1 &
CUDA_VISIBLE_DEVICES=2 python3 -W ignore train.py --lamda -5 --dataset_type cifar100 --lr 0.01 --alpha 2 --beta 0 --l2_reg 1e-2 --nr 0.4 --droGamma 0.5 --loss ABSCE --lamda_shots 80 --version 0706_ABSCE0.4_CIFAR100_0.9_lr_0.01 --data_path /home/qqi7/ABSGD/data --class_tau 1;


#TCE: nr 0.2:0.646, 0.6406, 0.6512 | nr 0.4: 0.5961,
CUDA_VISIBLE_DEVICES=0 python3 -W ignore train.py --lamda -3 --dataset_type cifar100 --lr 0.01 --alpha 6 --beta 0.5 --l2_reg 1e-4 --nr 0.2 --droGamma 0.9 --loss TCE --lamda_shots 80 --version 0706_TCE0.2_CIFAR100_0.9_lr_0.01 --data_path /home/qqi7/ABSGD/data;
CUDA_VISIBLE_DEVICES=0 python3 -W ignore train.py --lamda -3 --dataset_type cifar100 --lr 0.01 --alpha 6 --beta 0.5 --l2_reg 1e-4 --nr 0.4 --droGamma 0.9 --loss TCE --lamda_shots 80 --version 0706_TCE0.4_CIFAR100_0.9_lr_0.01 --data_path /home/qqi7/ABSGD/data;

# Reminder: cifar100 symmetric vs cifar100 asymmetric regularizer 不一样

#cifar100 asymmetric

#CE
CUDA_VISIBLE_DEVICES=2 python3 -W ignore train.py --lamda -3 --dataset_type cifar100 --lr 0.01 --alpha 0.2 --beta 2 --l2_reg 1e-4 --nr 0.4 --droGamma 0.9 --loss CE --lamda_shots 80 --version 0706_TCE0.4_CIFAR10_0.9_lr_0.01 --asym True --data_path /home/qqi7/ABSGD/data;


#SCE
CUDA_VISIBLE_DEVICES=3 python3 -W ignore train.py --lamda -3 --dataset_type cifar100 --lr 0.01 --alpha 0.2 --beta 2 --l2_reg 1e-4 --nr 0.2 --droGamma 0.9 --loss SCE --lamda_shots 80 --version 0706_TCE0.2_CIFAR10_0.9_lr_0.01 --asym True  --data_path /home/qqi7/ABSGD/data;
CUDA_VISIBLE_DEVICES=3 python3 -W ignore train.py --lamda -3 --dataset_type cifar100 --lr 0.01 --alpha 6 --beta 0.5 --l2_reg 1e-4 --nr 0.2 --droGamma 0.9 --loss SCE --lamda_shots 80 --version 0706_TCE0.2_CIFAR10_0.9_lr_0.01 --asym True  --data_path /home/qqi7/ABSGD/data;

CUDA_VISIBLE_DEVICES=3 python3 -W ignore train.py --lamda -1 --dataset_type cifar100 --lr 0.01 --alpha 0.6 --beta 0.6 --l2_reg 1e-4 --nr 0.2 --droGamma 0.9 --loss SCE --lamda_shots 80 --version 0706_TCE0.2_CIFAR10_0.9_lr_0.01 --asym True  --data_path /home/qqi7/ABSGD/data;

#ABSCE
CUDA_VISIBLE_DEVICES=2 python3 -W ignore train.py --lamda -3 --dataset_type cifar100 --lr 0.01 --alpha 0.2 --beta 2 --l2_reg 1e-4 --nr 0.2 --droGamma 0.9 --loss ABSCE --lamda_shots 80 --version 0706_TCE0.2_CIFAR10_0.9_lr_0.01 --asym True  --data_path /home/qqi7/ABSGD/data;
CUDA_VISIBLE_DEVICES=1 python3 -W ignore train.py --lamda -3 --dataset_type cifar100 --lr 0.01 --alpha 0.2 --beta 2 --l2_reg 1e-4 --nr 0.2 --droGamma 0.9 --loss ABSCE --lamda_shots 80 --version 0706_TCE0.2_CIFAR10_0.9_lr_0.01 --asym True  --data_path /home/qqi7/ABSGD/data --class_tau 1;
CUDA_VISIBLE_DEVICES=1 python3 -W ignore train.py --lamda -0.5 --dataset_type cifar100 --lr 0.01 --alpha 0.2 --beta 0.2 --l2_reg 1e-4 --nr 0.2 --droGamma 0.9 --loss ABSCE --lamda_shots 80 --version 0706_TCE0.2_CIFAR10_0.9_lr_0.01 --asym True  --data_path /home/qqi7/ABSGD/data --class_tau 1;

CUDA_VISIBLE_DEVICES=2 python3 -W ignore train.py --lamda -0.5 --dataset_type cifar100 --lr 0.01 --alpha 0.6 --beta 0.6 --l2_reg 1e-4 --nr 0.2 --droGamma 0.9 --loss ABSCE --lamda_shots 80 --version 0706_TCE0.2_CIFAR10_0.9_lr_0.01 --asym True  --data_path /home/qqi7/ABSGD/data --class_tau 1;

CUDA_VISIBLE_DEVICES=3 python3 -W ignore train.py --lamda -1 --dataset_type cifar100 --lr 0.01 --alpha 0.6 --beta 0.6 --l2_reg 1e-4 --nr 0.2 --droGamma 0.9 --loss ABSCE --lamda_shots 80 --version 0706_TCE0.2_CIFAR10_0.9_lr_0.01 --asym True  --data_path /home/qqi7/ABSGD/data --class_tau 1;

CUDA_VISIBLE_DEVICES=3 python3 -W ignore train.py --lamda -1 --dataset_type cifar100 --lr 0.01 --alpha 0.6 --beta 0.6 --l2_reg 1e-4 --nr 0.2 --droGamma 0.9 --loss ABSCE --lamda_shots 80 --version 0706_TCE0.2_CIFAR10_0.9_lr_0.01 --asym True  --data_path /home/qqi7/ABSGD/data;




#TCE
CUDA_VISIBLE_DEVICES=1 python3 -W ignore train.py --lamda -3 --dataset_type cifar100 --lr 0.01 --alpha 0.2 --beta 2 --l2_reg 1e-4 --nr 0.2 --droGamma 0.9 --loss TCE --lamda_shots 80 --version 0706_TCE0.2_CIFAR10_0.9_lr_0.01 --asym True  --data_path /home/qqi7/ABSGD/data;
CUDA_VISIBLE_DEVICES=2 python3 -W ignore train.py --lamda -3 --dataset_type cifar100 --lr 0.01 --alpha 0.2 --beta 2 --l2_reg 5e-3 --nr 0.4 --droGamma 0.9 --loss TCE --lamda_shots 80 --version 0706_TCE0.4_CIFAR10_0.9_lr_0.01 --asym True --data_path /home/qqi7/ABSGD/data;
