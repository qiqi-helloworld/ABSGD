__author__ = 'Qi'
# Created by on 6/8/22.

import torchvision.transforms as transforms
import models
import time, os, random, warnings, argparse, shutil
import torch, torch.nn.parallel, torch.optim
import torch.backends.cudnn as cudnn
from myDataLoader import get_iWildCam_train_test_val_DataLoader
from  mylosses import ABSGD, get_train_rule_hyperparameters
from utils import AverageMeter, accuracy, ml_f1_score
from progress.bar import Bar
from utils import saved_path_res, pred_class
import torch.nn as nn
import numpy as np
import pandas as pd
import torch.optim as optim
from collections import Counter
import wandb


parser = argparse.ArgumentParser(description='PyTorch Training')
parser.add_argument('--results_dir', metavar="RESULTS_DIR", default='./PMAI_TrainingResults', help = 'pic_results dir')
parser.add_argument('--save', metavar = 'SAVE',  default='',help='save folder')
parser.add_argument('--save_path', metavar = 'SAVEPATH',  default='',help='save folder')
parser.add_argument('--res_name', metavar = 'FILENAME',  default='',help='save file name')
parser.add_argument('--random_seed',  default=None, help='replicate seed')

parser.add_argument('--model', metavar='ARCH', default='resnet50')
parser.add_argument('--loss_type', default="CE", type=str,
                    choices=['focal', 'ldam', 'abldam', 'abfocal', 'abce', 'ce', 'nebce'], help='loss type')
parser.add_argument('--rand_number', default=0, type=int, help='fix random number for data sampling')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-b', '--batch_size', default=128, type=int,
                    metavar='N',
                    help='mini-batch size')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight_decay', default=0, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print_freq', default=100, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--gpus',  default='0', help = 'gpus used for training - e.g 0,1,2,3')
parser.add_argument('--lamda', type=float, default=10)
parser.add_argument('--gamma', type=float, default=1, help='smooth parameter of focal loss')
parser.add_argument('--drogamma', type=float, default=0.9, help='moving average parameter of ROBSGD')
parser.add_argument('--alpha', type=float, default=1, help='balance parameter of focal loss')
parser.add_argument('--lamda_shots', type=int, default=160, help='Number of epochs to decrease lamda')
parser.add_argument('--CB_shots', type=int, default=16, help='Number of epochs to apply Class-Balanced Weighting')

parser.add_argument('--beta', default=0.9999, type=float, help=" beta in Reweighting")
parser.add_argument('--num_classes', default=10, type=int, help="classes of different datasets")
parser.add_argument('--cls_num_list', default=None, help="# of class distributions")
parser.add_argument('--frozen_aside_fc', default=False, type=eval, choices=[True, False],
                    help='whether frozen the feature layers (First three block)')
parser.add_argument('--not_frozen_last_block', default=False, type=eval, choices=[True, False],
                    help='whether frozen the feature layers (First three block)')

parser.add_argument('--robAlpha', default=0.5, type=float, help='Normalization Parameter for the Normalization Term')
parser.add_argument('--isTau', default=False, type=eval, choices=[True, False],
                    help='Whether Normalize the calssifier layer.')
parser.add_argument('--use_BN', default=False, type=eval, choices=[True, False],
                    help='Whether use BN before the fully connected layer.')
parser.add_argument('--train_defer', default=1, type=int, help='defer or not')
parser.add_argument('--DP', default=0, type=float, help='value of percentage of save samples in drop out')
parser.add_argument('--class_tau', default=0, type=float, help="# adaptive normalization for softamx")
parser.add_argument('--stage', default=1, type = int, help = "which stage are you in by myself.")
parser.add_argument('--type', default='torch.cuda.FloatTensor',
                    help = 'types of tensor - e.g torch.cuda.FloatTensor')
parser.add_argument('--lr_schedule', default='stepLR', type=str, help = "training straties")
parser.add_argument('--u', default=0, type=float, help = "the average moving stochastic estimator ")
parser.add_argument('--works', default=8, type=int, help = 'number of threads used for loading data')
parser.add_argument('--dataset', type = str, default='iwildcam')

parser.add_argument('--is_cls', type = bool,  default=False)
parser.add_argument('--is_feat_frozen', type = bool,  default=False, help = 'whether frozen pretrained feature layers')
parser.add_argument('--sc_gamma', type = float,  default=0.1, help = 'stagewise decay rate ')
parser.add_argument('--truncate_weight', type = float,  default=20, help = 'stagewise decay rate in ')



def train(args, train_loader, model, criterion, optimizer, myLambda, epoch):
    bar = Bar('Processing', max=len(train_loader))
    batch_time = AverageMeter('batch time')
    data_time = AverageMeter('time')
    losses = AverageMeter('losses')
    top1 = AverageMeter('top1')
    top5 = AverageMeter('top5')

    model.train()
    # Train loop
    start_batch_time = time.time()
    i = 0

    cls_weight = None  # standard ROBSGD
    if args.is_cls:
        cls_weight, _ = get_train_rule_hyperparameters(args, train_rule='reweight', epoch=epoch)


    for labeled_batch in zip(train_loader):
        data_time.update(time.time() - start_batch_time)
        if i%1000 == 0:
            print(i, len(train_loader))
        (inputs, targets, metadata) = labeled_batch[0]
        inputs, targets, metadata = inputs.cuda(), targets.cuda(), metadata.cuda()
        outputs = model(inputs)
        # print(">>>>", outputs.size())
        # loss = criterion(outputs, targets, None, myLambda)
        # print(cls_weight)
        outputs = outputs / (torch.norm(outputs, p=2, dim=1, keepdim=True) ** args.class_tau)
        loss = criterion(outputs, targets, cls_weights = cls_weight, myLambda = myLambda)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # print(outputs.size(), targets.size())
        # print(metadata, targets)

        bth_top1, bth_top5 = accuracy(outputs, targets, topk=(1, 3))
        losses.update(loss.item(), outputs.size(0))
        top1.update(bth_top1[0].item(), outputs.size(0))
        top5.update(bth_top5[0].item(), outputs.size(0))

        # plot progress
        bar.suffix = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | top1: {top1: .4f}'.format(
            batch= i + 1,
            size=len(train_loader),
            data=data_time.avg,
            bt=batch_time.avg,
            total=bar.elapsed_td,
            eta=bar.eta_td,
            loss=losses.avg,
            top1=top1.avg,
        )
        bar.next()
        i+=1
    bar.finish()

    batch_time.update(time.time() - start_batch_time)

    return losses.avg, top1.avg, top5.avg

def validate(args, val_loader, model, epoch, split_name = 'id_val'):
    bar = Bar('Processing', max=len(val_loader))
    batch_time = AverageMeter('batch time')
    data_time = AverageMeter('time')
    losses = AverageMeter('losses')
    top1 = AverageMeter('top1')
    top5 = AverageMeter('top5')

    # switch to evaluate mode
    model.eval()
    CE_criterion = nn.CrossEntropyLoss().cuda()

    with torch.no_grad():
        start_batch_time = time.time()
        all_outputs = torch.Tensor([]).cuda()
        all_targets = torch.Tensor([]).cuda()

        i = 0
        all_pred_class = []
        all_pred_class_prob = []
        for labeled_batch in enumerate(val_loader):

            # if i>=5:
            #       break
            # print( type(labeled_batch), len(labeled_batch[1]))
            (inputs, targets, metadata) = labeled_batch[1]
            # measure data loading time
            data_time.update(time.time() - start_batch_time)
            targets = targets.cuda(non_blocking = True)
            inputs = inputs.cuda(non_blocking = True)
            # compute output
            outputs = model(inputs)
            # measure accuracy and record loss
            all_outputs = torch.cat([all_outputs, outputs], dim=0)
            all_targets = torch.cat([all_targets, targets], dim=0)
            loss = CE_criterion(outputs, targets)

            bth_top1, bth_top5 = accuracy(outputs.data, targets, topk=(1, 5))
            top1.update(bth_top1[0].item(), outputs.size(0))
            top5.update(bth_top5[0].item(), outputs.size(0))
            losses.update(loss.mean().item(), outputs.size(0))

            all_pred_class.extend(pred_class(outputs.cpu()).numpy().tolist())
            all_pred_class_prob.extend(outputs.cpu().numpy().tolist())



            # print(loss_avg, prec1_avg)
            # measure elapsed time
            batch_time.update(time.time() - start_batch_time)
            i+= 1
            # plot progress
            bar.suffix = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | top1: {top1: .4f}'.format(
                batch=i + 1,
                size=len(val_loader),
                data=data_time.avg,
                bt=batch_time.avg,
                total=bar.elapsed_td,
                eta=bar.eta_td,
                loss=losses.avg,
                top1=top1.avg,
            )
            bar.next()
            i+=1

        bar.finish()
        # print('loss is {loss} and top1 is {top1}'.format(loss = losses.avg, top1=top1.avg))

        f1 = ml_f1_score(all_targets.cpu().numpy().tolist(), all_pred_class)
        cur_res_dir = args.dataset + '_lr_' + str(args.lr) + '_cls_tau_' + str(args.class_tau) + '_epochs_' + str(args.epochs) + '_is_cls_' + str(args.is_cls) + '_lamda_' + str(args.lamda) + '_TruncatW_' + str(args.truncate_weight) +'_epoch_' + str(args.epochs) + '_bth_' + str(args.batch_size)
        if not os.path.exists(os.path.join('./results/iwildcam/', cur_res_dir)):
            os.mkdir(os.path.join('./results/iwildcam', cur_res_dir))
        file_name = './results/iwildcam/{cur_res_dir}/{dataset}_split:{split}_seed:{seed}_epoch_{epoch}.csv'.format(dataset = args.dataset, cur_res_dir = cur_res_dir, split =split_name , seed = args.random_seed, epoch = epoch)
        prob_file_name = './results/iwildcam/{cur_res_dir}/prob_{dataset}_split:{split}_seed:{seed}_epoch_{epoch}.csv'.format(dataset = args.dataset, cur_res_dir = cur_res_dir, split =split_name , seed = args.random_seed, epoch = epoch)
        # print(all_pred_class)
        pd_pred_class = pd.DataFrame(all_pred_class)
        pd_all_pred_class_prob = pd.DataFrame(all_pred_class_prob)
        pd_pred_class.to_csv(file_name, header=False, index=False)
        pd_all_pred_class_prob.to_csv(prob_file_name, header=False, index=False)




    return losses.avg, top1.avg, top5.avg, f1, './results/iwildcam/{cur_res_dir}/'.format(cur_res_dir = cur_res_dir)


def save_checkpoint(state, is_best, checkpoint='checkpoint', filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'model_best.pth.tar'))




def main():
    args = parser.parse_args()
    global best_prec1, z_t
    z_t = dict()
    # args.CB_shots = args.epochs // 2
    wandb.init(config=args, project="wilds")
    overall_training_time = 0
    ######VERBOSE
    print(args)
    best_prec1 = 0
    results = saved_path_res(args)
    if args.random_seed is not None:
        torch.manual_seed(args.random_seed)
        random.seed(args.random_seed)
        cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if 'cuda' in args.type:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
        args.gpus = [int(i) for i in args.gpus.split(',')]
        cudnn.benchmark = True
        print("Use GPU: {} for training".format(args.gpus))
    else:
        args.gpus = None

    # create model
    print("=> creating model '{}'".format(args.model))
    train_loader, val_loader, test_loader, id_val_loader, id_test_loader = get_iWildCam_train_test_val_DataLoader(args.batch_size)
    args.num_classes = len(np.unique(train_loader.dataset.y_array.numpy()))
    # print("Models Arch:", args.model)
    model = models.__dict__[args.model](pretrained = True)
    model.fc = nn.Linear(2048, args.num_classes)


    if args.is_feat_frozen:
        network_frozen(args, model)
    if args.gpus and len(args.gpus) >= 1:
        print("We are running the model in GPU :", args.gpus)
        model = torch.nn.DataParallel(model)
        model.type(args.type)

    optimizer = torch.optim.Adam(model.parameters(), args.lr, weight_decay=args.weight_decay)
    # optimizer = torch.optim.SGD(model.parameters(), momentum=0.9, lr =  args.lr, weight_decay=args.weight_decay)

    criterion = ABSGD(args, 'CE', abAlpha = 1)
    # criterion = nn.CrossEntropyLoss()

    class_num_dict = Counter(train_loader.dataset.y_array.numpy())
    sorted_class_num_dict = sorted(class_num_dict.items(), key=lambda x: x[0])
    args.cls_num_list = torch.tensor(np.array(sorted_class_num_dict)[:, 1])
    args.cls_num_list[args.cls_num_list >=args.truncate_weight] = args.truncate_weight


    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.epochs//2, gamma=args.sc_gamma)

    start_time = time.time()
    for epoch in range(args.epochs):


        print('train_loader :', len(train_loader))
        if epoch < args.epochs//2:
            myLambda = 200
        else:
            myLambda = args.lamda
        train_loss, train_top1, train_top5 = train(args, train_loader, model, criterion, optimizer, myLambda, epoch)


        print('val_loader :', len(val_loader))
        val_loss, val_top1, val_top5, val_f1, _ = validate(args, val_loader, model, epoch, split_name='val')

        print('test_loader :', len(test_loader))
        test_loss, test_top1, test_top5, test_f1, _= validate(args, test_loader, model, epoch, split_name='test')

        print('val_loader :', len(val_loader))
        id_val_loss, id_val_top1, id_val_top5, id_val_f1, _ = validate(args, id_val_loader, model, epoch, split_name='id_val')

        print('test_loader :', len(test_loader))
        id_test_loss, id_test_top1, id_test_top5, id_test_f1, res_dir = validate(args, id_test_loader, model, epoch, split_name='id_test')
        scheduler.step()

        print((time.time() -start_time)//60, 'minutes')
        best_prec1 = max(test_top1, val_top1 if test_loader is None else test_top1)
        print('\n Epoch: {0}\t'
              'Training Loss {train_loss:.4f} \t'
              'Training Prec@1 {train_prec1:.3f} \t'
              'Validation Loss {val_loss:.4f} \t'
              'Validation Prec@1 {val_prec1:.3f} \t'
              'Test Prec@1 {test_prec1:.3f}\t'
         'ID_Test Prec@1 {id_test_prec1:.3f}\t'
              'Best Test Prec@1 {best_test_prec1:.3f} \t'
              .format(epoch + 1, train_loss=train_loss, val_loss=val_loss,
                      train_prec1=train_top1, val_prec1=val_top1,
                      test_prec1=test_top1 if test_loader is not None else val_top1,
                      best_test_prec1=best_prec1, id_test_prec1 = id_test_top1))

        print('F1 Metrics \t'
          'Val f1 {val_f1:.3f}\t'
          'Test f1 {test_f1:.3f}\t'
          'ID_Val f1 {id_val_f1:.3f}\t'
          'ID_Test f1 {id_test_f1:.3f}\t'.format(val_f1 = val_f1, test_f1 = test_f1, id_val_f1 = id_val_f1, id_test_f1 = id_test_f1))

        results.add(epoch= epoch + 1, train_loss=train_loss, val_loss=val_loss,
                    train_prec1=train_top1, val_prec1=val_top1,
                    test_prec1=test_top1 if test_loader is not None else val_top1, best_test_prec1=best_prec1,
                    train_prec5=train_top5, val_prec5=val_top5, id_val_top1 = id_val_top1, id_test_top1 = id_test_top1, val_f1 = val_f1, test_f1 = test_f1, id_val_f1 = id_val_f1, id_test_f1 = id_test_f1)

        wandb.log({"train_loss" :train_loss, "val_loss":val_loss,
                    "train_prec1":train_top1, "val_prec1":val_top1,
                   "test_prec1":test_top1 if test_loader is not None else val_top1, "best_test_prec1":best_prec1,
                    "train_prec5":train_top5, "val_prec5":val_top5, "id_val_top1" : id_val_top1, "id_test_top1" : id_test_top1,
                   'val_f1':val_f1, 'test_f1':test_f1, 'id_val_f1':id_val_f1, 'id_test_f1':id_test_f1}, step = epoch)

        save_checkpoint(model.state_dict(), False, checkpoint=res_dir, filename=str(epoch)+ '_epoch.pth.tar')

    print('args.cls_num_list: ', args.cls_num_list, len(args.cls_num_list[args.cls_num_list<100]))
    print('Cost Total number pf {minutes}'.format(minutes = (time.time() - start_time)//60))



def network_frozen(args, model):
    last_block_number = 2

    last_block_pattern = 'layer4.' + str(last_block_number)

    total_layers = 0
    for param_name, param in model.named_parameters():  # (self.networks[key]):  # frozen the first 3 block
        total_layers +=1
        if 'fc' not in param_name and "linear" not in param_name:
            param.requires_grad = False
            # if args.not_frozen_last_block:
            #     if last_block_pattern in param_name:
            #         param.requires_grad = True

    cnt_layers = 0
    for param_name, param in model.named_parameters():
        if param.requires_grad:
            cnt_layers += 1
            # print(param_name)
    print("{0}/{1} number of trained layers".format(cnt_layers, total_layers))


if __name__ == '__main__':
    main()

