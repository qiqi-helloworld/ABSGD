__author__ = 'Qi'
# Created by on 9/5/22.
import pandas as pd
import torch


def pred_class(output, topk=(1,)):
    with torch.no_grad():
        maxk = max(topk)
        _, pred = output.topk(maxk, 1, True, True)

    return pred.view(-1)


epoch = 19

dat_1 = pd.read_csv('./prob_iwildcam_split:test_seed:1_epoch_' +str(epoch) + '.csv', header=None).values
dat_2 = pd.read_csv('./prob_iwildcam_split:test_seed:1_epoch_' +str(epoch+1) + '.csv',  header=None).values
dat_3 = pd.read_csv('./prob_iwildcam_split:test_seed:1_epoch_' +str(epoch) + '.csv', header=None).values

# print(dat_1.head())

dat = (dat_1+dat_2+ dat_3)/2

class_dat = pred_class(torch.tensor(dat))
print(dat.shape, class_dat.shape, pd.DataFrame(class_dat).shape)
pd.DataFrame(class_dat).to_csv('./ensemble/iwildcam_split:test_seed:2_epoch_'+str(epoch) + '_4e-05_cls_0.csv', header=False, index=False)

#
# dat_1 = pd.read_csv('./prob_iwildcam_split:id_test_seed:0_epoch_' +str(epoch) + '.csv',  header=None).values
# dat_2 = pd.read_csv('./prob_iwildcam_split:id_test_seed:1_epoch_' +str(epoch) + '.csv',  header=None).values
# dat_3 = pd.read_csv('./prob_iwildcam_split:id_test_seed:2_epoch_' +str(epoch) + '.csv',  header=None).values
#
# # print(dat_1.head())
#
# dat = (dat_1+dat_2 + dat_3)/3
# class_dat = pred_class(torch.tensor(dat))
# print(dat.shape, class_dat.shape, pd.DataFrame(class_dat).shape)
# pd.DataFrame(class_dat).to_csv('./ensemble/iwildcam_split:id_test_seed:2_epoch_'+str(epoch) + '_4e-05_cls_0.csv', header=False, index=False)
#
#
#
# dat_1 = pd.read_csv('./prob_iwildcam_split:val_seed:0_epoch_' + str(epoch) + '.csv', header=None).values
# dat_2 = pd.read_csv('./prob_iwildcam_split:val_seed:1_epoch_' +str(epoch) + '.csv',  header=None).values
# dat_3 = pd.read_csv('./prob_iwildcam_split:val_seed:2_epoch_' +str(epoch) + '.csv',  header=None).values
#
# # print(dat_1.head())
#
# dat = (dat_1+dat_2 + dat_3)/3
#
# class_dat = pred_class(torch.tensor(dat))
# print(dat.shape, class_dat.shape, pd.DataFrame(class_dat).shape)
# pd.DataFrame(class_dat).to_csv('./ensemble/iwildcam_split:val_seed:2_epoch_'+str(epoch) + '_4e-05_cls_0.csv', header=False, index=False)
#
# dat_1 = pd.read_csv('./prob_iwildcam_split:id_val_seed:0_epoch_' + str(epoch) + '.csv',  header=None).values
# dat_2 = pd.read_csv('./prob_iwildcam_split:id_val_seed:1_epoch_' + str(epoch) + '.csv',  header=None).values
# dat_3 = pd.read_csv('./prob_iwildcam_split:id_val_seed:2_epoch_' + str(epoch) + '.csv',  header=None).values
#
# # print(dat_1.head())
#
# dat = (dat_1+dat_2 + dat_3)/3
# class_dat = pred_class(torch.tensor(dat))
# print(dat.shape, class_dat.shape, pd.DataFrame(class_dat).shape)
# pd.DataFrame(class_dat).to_csv('./ensemble/iwildcam_split:id_val_seed:2_epoch_'+str(epoch) + '_4e-05_cls_0.csv', header=False, index=False)
