
Experiemental Results Replicate
----------------------------------
```python
/home/qiuzh/.conda/envs/vlp/bin/python -W ignore main.py --batch_size 16 --lr 4e-5 --gpus 5 --random_seed 0 --epochs 36 --CB_shots 5 --truncate_weight 10 --lamda 1.1 --sc_gamma 0.5 --is_cls True;
/home/qiuzh/.conda/envs/vlp/bin/python -W ignore main.py --batch_size 16 --lr 4e-5 --gpus 6 --random_seed 1 --epochs 36 --CB_shots 5 --truncate_weight 10 --lamda 1.1 --sc_gamma 0.5 --is_cls True;
/home/qiuzh/.conda/envs/vlp/bin/python -W ignore main.py --batch_size 16 --lr 4e-5 --gpus 7 --random_seed 2 --epochs 36 --CB_shots 5 --truncate_weight 10 --lamda 1.1 --sc_gamma 0.5 --is_cls True;
```



Results Post Process
--------------------------------
To further boosting the results, we average the prediction probability of different runs for every epoch and predict the class. Then pick the best three class prediction csv files in terms of OOD test F1 values as the final submission.
