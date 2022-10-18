__author__ = 'Qi'
# Created by on 1/31/22.
import argparse
import os
import random
import warnings
from sklearn.exceptions import DataConversionWarning
warnings.filterwarnings(action='ignore')

import sys
import numpy as np
import torch, torch.nn.parallel, torch.optim
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.multiprocessing as mp
import torch.utils.data
import random
import models
# from tensorboardX import SummaryWriter
from sklearn.metrics import confusion_matrix
from myutils import accuracy, saved_path_res, AverageMeter, save_checkpoint_epoch, get_tsne_of_sample_feature, adjust_learning_rate, network_frozen, loaded_pretrained_models, save_best_checkpoint_epoch, get_weights_of_majority_minority_class, get_wieghts_of_each_class, get_train_rule_hyperparameters
from myDatasets import get_num_classes, get_cls_num_list
from myDataLoader import get_train_val_test_loader
from mylosses import get_train_loss
import time
import wandb
import pandas as pd
#import matplotlib.pyplot as plt

# def my_warn():
#     pass
# warnings.warn = my_warn


model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch Cifar Training')
parser.add_argument('--results_dir', metavar="RESULTS_DIR", default='./PMAI_TrainingResults', help = 'pic_results dir')
parser.add_argument('--save', metavar = 'SAVE',  default='',help='save folder')
parser.add_argument('--dataset', default='cifar10', help='dataset setting')
parser.add_argument('--model', metavar='ARCH', default='resnet32',
                    choices=model_names,
                    help='model architecture: ' +
                         ' | '.join(model_names) +
                         ' (default: resnet32)')
parser.add_argument('--loss_type', default="CE", type=str,
                    choices=['focal', 'ldam', 'robldam', 'robfocal', 'robce', 'ce', 'nebce'], help='loss type')
parser.add_argument('--imb_type', default="exp", type=str, help='imbalance type')
parser.add_argument('--imb_factor', default=0.01, type=float, help='imbalance factor')
parser.add_argument('--train_rule', default='None', type=str, choices=['None', 'resample', 'reweight'],
                    help='data sampling strategy for train loader')
parser.add_argument('--rand_number', default=0, type=int, help='fix random number for data sampling')
parser.add_argument('--exp_str', default='0', type=str, help='number to indicate which experiment it is')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start_epochs', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch_size', default=128, type=int,
                    metavar='N',
                    help='mini-batch size')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight_decay', default=2e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print_freq', default=100, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', default=False, type=eval, choices=[True, False],
                    help='use pre-trained model')
parser.add_argument('--topK', default=None, type=int,
                    help='use pre-trained model')

parser.add_argument('--seed', default=0, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpus',  default='0', help = 'gpus used for training - e.g 0,1,2,3')
parser.add_argument('--root_log', type=str, default='log')
parser.add_argument('--store_name', type=str, default='log')
parser.add_argument('--root_model', type=str, default='checkpoint')
parser.add_argument('--lamda', type=float, default=1)
parser.add_argument('--neb_tau', type=float, default=1, help = 'soft neighboring parameter')
parser.add_argument('--init_lamda', type=float, default=10)
parser.add_argument('--start_time', type=float, default=100)
parser.add_argument('--repeats', type=int, default=0)
parser.add_argument('--alg', type=str, help='Algorithm')
parser.add_argument('--gamma', type=float, default=1, help='smooth parameter of focal loss')
parser.add_argument('--drogamma', type=float, default=0.1, help='moving average parameter of ROBSGD')
parser.add_argument('--alpha', type=float, default=1, help='balance parameter of focal loss')
parser.add_argument('--RENORM', default=True, type=eval, choices=[True, False],
                    help='Renormalized MSCGD or MSCGD')
parser.add_argument('--lamda_shots', type=int, default=160, help='Number of epochs to decrease lamda')
parser.add_argument('--CB_shots', type=int, default=60, help='Number of epochs to apply Class-Balanced Weighting')

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
parser.add_argument('--ngroups', default=1, type=int, help='number of groups in a minibatch')
parser.add_argument('--option', default='I', type=str, help='Group Choice')
parser.add_argument('--train_defer', default=1, type=int, help='defer or not')
parser.add_argument('--DP', default= 0, type=float, help='value of percentage of save samples in drop out')
parser.add_argument('--class_tau', default=0, type=float, help="# adaptive normalization for softamx")
parser.add_argument('--frozen_start', default=160, type=int,
                    help='# number of epochs that start to frozen the feature layers.')
parser.add_argument('--stage', default=1, type = int, help = "which stage are you in by myself.")
parser.add_argument('--type', default='torch.cuda.FloatTensor',
                    help = 'types of tensor - e.g torch.cuda.FloatTensor')
parser.add_argument('--lr_schedule', default='stepLR', type=str, help = "training straties")
parser.add_argument('--u', default=0, type=float, help = "the average moving stochastic estimator ")
parser.add_argument('--res_name', default=None, type=str, help = "results name of file")
parser.add_argument('--works', default=8, type=int, help = 'number of threads used for loading data')


best_acc1 = 0
def main():

    args = parser.parse_args()
    global best_prec1, z_t
    z_t = dict()
    wandb.init(config=args, project="noisy_labels", entity="qiqi-helloworld")
    overall_training_time = 0
    ######VERBOSE
    print(args)

    print('drogamma :', args.drogamma )
    save_path, results = saved_path_res(args)
    if args.seed is not None:
        torch.manual_seed(args.seed)
        random.seed(args.seed)
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
    use_norm = True if 'ldam' in args.loss_type else False
    args.num_classes = get_num_classes(args)

    # args.cls_num_list = get_cls_num_list(args)
    # np.save('places_LT.npy', args.cls_num_list)

    print('>>>>>>>>>>>>>> :', args.cls_num_list)
    print("Models Arch:", args.model)
    if 'cifar' in args.dataset:
        model = models.__dict__[args.model](num_classes=args.num_classes, use_norm=use_norm, DP = args.DP)
    elif 'imagenet' in args.dataset:
        feat_dim = 2048
        use_fc_add = False
        model = models.__dict__[args.model](args.num_classes, pretrained=args.pretrained, data=args.dataset, \
                                            dropout=args.DP, use_BN=args.use_BN, isTau=args.isTau,
                                            use_fc_add=use_fc_add, feat_dim=feat_dim)
    elif 'places' in args.dataset:
        use_fc_add = False
        if args.stage == 2:
            feat_dim = 2048
        elif args.stage == 3:
            use_fc_add = True
            feat_dim = 512

        model= models.__dict__[args.model](args.num_classes, pretrained = args.pretrained, data = args.dataset,\
                      dropout = args.DP,  use_BN = args.use_BN, isTau = args.isTau, use_fc_add = use_fc_add, feat_dim = feat_dim )


    if args.gpus and len(args.gpus) >= 1:
        print("We are running the model in GPU :", args.gpus)
        model = torch.nn.DataParallel(model)
        model.type(args.type)

    # Load check points from certain number of epochs.
    if args.pretrained:
        loaded_pretrained_models(args, model)
        print("Pretrained Model Loaded Success.")

    train_cls = []
    train_loader, val_loader, test_loader = get_train_val_test_loader(args, train_sampler=None)
    if args.cls_num_list is None:
        args.cls_num_list = get_cls_num_list(args)

    criterion = get_train_loss(args, args.loss_type)
    optimizer = torch.optim.SGD(model.parameters(), args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    if args.lr_schedule == 'coslr': # learning rates: coslr or stepLR
        print("we are using CosineAnnealingLR")
        args.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, args.epochs, eta_min=0)
        print("Initial Learning rates for the first epochs:", args.scheduler.get_lr())
    if args.frozen_aside_fc:
        print("We are just training part of the neural network")
        network_frozen(args, model)
        print("frozen finished")

    test_loss, best_prec1, training_time, best_epoch = 0, 0, 0, 0
    CE_criterion = nn.CrossEntropyLoss(reduction='none')
    if test_loader is not None:
        _, pretrain_val_prec1, _, _, _, _, _, _, _ = validate(args, test_loader, model, CE_criterion, 0, optimizer, args.init_lamda, None)
        print("pretrain_testl_prec1  {:.4f}".format(pretrain_val_prec1))
    else:
        _, pretrain_val_prec1, _, _, _, _, _, _, _= validate(args, val_loader, model, CE_criterion, 0, optimizer, args.init_lamda, None)
        print("pretrain_val_prec1  {:.4f}".format(pretrain_val_prec1))

    # plt.figure()

    # get_tsne_of_sample_feature(args, train_loader, model, 'train', 0)
    # get_tsne_of_sample_feature(args, val_loader, model, 'val', 0)
    # get_tsne_of_sample_feature(args, val_loader, model, 'test', 0)
    for epoch in range(args.start_epochs, args.epochs):
        # adjust learning rates
        if args.lr_schedule == 'stepLR':
            adjust_learning_rate(optimizer, epoch, args)
        if args.lr_schedule == 'coslr':
            args.scheduler.step()
        print("lr : ", optimizer.param_groups[0]['lr'])


        # this requires the knowledge from https://openaccess.thecvf.com/content_CVPR_2019/papers/Cui_Class-Balanced_Loss_Based_on_Effective_Number_of_Samples_CVPR_2019_paper.pdf
        # train
        cls_weights, _ = get_train_rule_hyperparameters(args, args.train_rule, epoch)
        myLambda = args.init_lamda
        if epoch >= args.lamda_shots:
            myLambda = args.lamda
        _, _, _, epoch_training_time, _, _, _, _, _ = train(
            args, train_loader, model, criterion, epoch, optimizer, myLambda, cls_weights)
        training_time += epoch_training_time

        # if args.lr_schedule == 'coslr':
        #     wandb.log({"lr": args.scheduler.get_lr(), 'lambda': myLambda}, step = epoch)
        # else:
        #     wandb.log({"lr": lr, 'lambda': myLambda}, step = epoch)
        # print("epoch: ", epoch, " | lr:", optimizer.param_groups[0]['lr'])

        train_loss, train_prec1, train_prec5, _, majority_train_loss, minority_train_loss, majority_P, minority_P, cls_p = validate(args, train_loader, model, CE_criterion, epoch, optimizer, myLambda, None)
        val_loss, val_prec1, val_prec5, _, _, _, _, _, _ = validate(
            args, val_loader, model, CE_criterion, epoch, optimizer, myLambda, None)
        if test_loader is not None:
            test_loss, test_prec1, test_prec5, _, _, _, _, _, _ = validate(
                args, test_loader, model, CE_criterion, epoch, optimizer, myLambda, None)

        # if epoch >= args.lamda_shots:
        train_cls.append(cls_p)
        # if epoch == args.lamda_shots or epoch == args.epochs-1 or epoch == 180:
        #     # print(cls_p)
        #     # plt.plot(cls_p, label = str(epoch))


        tmp_prec1 = val_prec1 if test_loader is None else test_prec1
        is_best = tmp_prec1 > best_prec1
        print(">>>>>>>>>>>>> :", is_best, ": <<<<<<<<<<<<<<")
        if is_best:
            best_prec1 = tmp_prec1
            best_epoch = epoch
        if is_best:
            save_best_checkpoint_epoch({
                 'epoch': epoch,
                 'model': args.model,
                 'batch_size': args.batch_size,
                 'state_dict': model.module.state_dict(),
                 'time': overall_training_time
             }, is_best=is_best, path=save_path)

        print('\n Epoch: {0}\t'
                     'Training Loss {train_loss:.4f} \t'
                     'Training Prec@1 {train_prec1:.3f} \t'
                     'Validation Loss {val_loss:.4f} \t'
                     'Validation Prec@1 {val_prec1:.3f} \t'
                     'Test Prec@1 {test_prec1:.3f}\t'
                     'Best Test Prec@1 {best_test_prec1:.3f} \t'
                     'Best Epoch {best_epoch:.3f} \t'
                     .format(epoch + 1, train_loss=train_loss, val_loss=val_loss,
                             train_prec1=train_prec1, val_prec1=val_prec1, test_prec1=test_prec1 if test_loader is not None else val_prec1,
                             best_test_prec1=best_prec1, best_epoch = best_epoch))
        results.add(epoch=epoch + 1, train_loss=train_loss, val_loss=val_loss,
                    train_prec1=train_prec1, val_prec1=val_prec1, test_prec1=test_prec1 if test_loader is not None else val_prec1, best_test_prec1=best_prec1,
                    train_prec5=train_prec5, val_prec5=val_prec5, best_epoch = epoch, overall_training_time=overall_training_time // 60)

        results.save()

        wandb.log({"train loss": train_loss, 'train acc1': train_prec1, 'train acc5': train_prec5}, step=epoch)
        wandb.log({"majority train loss": majority_train_loss, "minority train loss": minority_train_loss}, step=epoch)
        wandb.log({"majority P": majority_P, "minority P": minority_P}, step=epoch)
        #
        wandb.log({"best test acc": best_prec1}, step=epoch)
        if test_loader is not None:
            wandb.log({"test loss": test_loss, 'test acc1': test_prec1, 'test acc5': test_prec5}, step=epoch)
        else:
            wandb.log({"test loss": val_loss, 'test acc1': val_prec1, 'test acc5': val_prec5}, step=epoch)

    # plt.title(args.dataset+' ' + args.imb_type)
    # plt.legend()
    # plt.savefig('_'.join([args.dataset, args.imb_type , str(args.imb_factor), 'clsp.png']))

    pd.DataFrame(train_cls).to_csv(args.root_log + '/' + args.res_name + '_train_clsp_0615.csv', header=None, index=False)
    if args.lamda >= 200:
        print("We use the method of SGD")
    else:
        print("We implement DRO with lambd : ", args.lamda)


def forward(args, data_loader, model, criterion, epoch, optimizer, cls_weights, myLambda = 0, training=True):

    run_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    majority_losses = AverageMeter('Loss', ':.4e')
    minority_losses = AverageMeter('Loss', ':.4e')
    majority_P = AverageMeter('P', ':.4e')
    minority_P = AverageMeter('P', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    covid_top1 = AverageMeter('COVID-Acc@1', ':6.2f')
    end = time.time()
    majP, minP = 0, 0
    cls_p = None
    if training:
        # im_id, tax_id
        for i, (inputs, targets)  in enumerate(data_loader):
            # measure data loading time
            run_time.update(time.time() - end)
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs, feat = model(inputs)
            # print(output.size())
            # norm1 = torch.norm(output, p = 2, dim = 1, keepdim= True)
            # norm001 = torch.norm(output, p = 2, dim = 1, keepdim= True)**args.class_tau
            # print('Sum of output: %.3f'%torch.sum(output), 'L2 norm: %.3f'%norm1[0].item(), 'CLASS level norm: %.3f'%norm001[0].item())
            # print(">>>>>>>>>>>> :", outputs.size(), ": <<<<<<<<<<<<<<<<<")
            # print(outputs)
            outputs = outputs/(torch.norm(outputs, p = 2, dim = 1, keepdim= True)**args.class_tau)
            # print(output.size(), torch.sum(output, dim= 1, keepdim=True)[0], torch.norm(output, p=2, dim=1, keepdim= True)[0][0],  torch.norm(output,dim=1, keepdim= True)[0])

            # loss
            if 'rob' in args.loss_type:
                loss = criterion(outputs, targets, cls_weights, myLambda)
                args.u  = criterion.u
            elif 'neb' in args.loss_type:
                loss = criterion(outputs, targets, cls_weights, myLambda, feat)
            else:
                loss = criterion(outputs, targets, cls_weights)

            # accuracy
            if args.num_classes >= 5:
                acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
            else:
                acc1, acc5 = accuracy(outputs, targets, topk=(1, 3))

            losses.update(loss.item(), inputs.size(0))
            top1.update(acc1[0].item(), inputs.size(0))
            top5.update(acc5[0].item(), inputs.size(0))

            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            run_time.update(time.time() - end)
            end = time.time()
            if i % args.print_freq == 0:
                print('{phase} - Epoch: [{0}][{1}/{2}]\t'
                            'Data {run_time.sum:.3f} ({run_time.val:.3f})\t'
                            'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                             'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                    epoch, i, len(data_loader),
                phase='TRAINING' if training else 'EVALUATING',
                    run_time=run_time, loss=losses, top1=top1))
                # print(">>>:", criterion.u, ":<<<")


    else:
        all_preds = []
        all_targets = []
        all_losses = []
        # print('args.u : ', args.u)
        for i, (inputs, targets) in enumerate(data_loader):

            inputs, targets = inputs.cuda(), targets.cuda()
            outputs, _ = model(inputs)
            loss = criterion(outputs, targets)


            majority_loss = loss[targets < args.num_classes//2]
            minority_loss = loss[targets >= args.num_classes//2]

            if type(outputs) is list:
                outputs = outputs[0]
            if args.num_classes >= 5:
                prec1, prec5 = accuracy(outputs.data, targets, topk=(1, 5))
            else:
                prec1, prec5 = accuracy(outputs.data, targets, topk=(1, 3))

            losses.update(loss.mean().item(), inputs.size(0))


            if args.u != 0:
                majP, minP = get_weights_of_majority_minority_class(args, loss, targets, args.u, myLambda)
                # print(majP, minP)

            if majority_loss.size(0) == 0:
                majority_losses.update(0, majority_loss.size(0))
                majority_P.update(0, majority_loss.size(0))
            else:
                majority_losses.update(majority_loss.mean().item(), majority_loss.size(0))
                majority_P.update(majP, majority_loss.size(0))


            if minority_loss.size(0) == 0:
                minority_losses.update(0, minority_loss.size(0))
                minority_P.update(0, minority_loss.size(0))
            else:
                minority_losses.update(minority_loss.mean().item(), minority_loss.size(0))
                minority_P.update(minP, minority_loss.size(0))

            # print(">>>>>>>>>>>>>:", minority_loss.size(), ">>>>>>>>>>>>>:",)
            top1.update(prec1[0].item(), inputs.size(0))
            top5.update(prec5[0].item(), inputs.size(0))
            run_time.update(time.time() - end)
            end = time.time()
            if i % args.print_freq == 0 :
                print('{phase} - Epoch: [{0}/{1}][{2}/{3}]\t'
                             'Data {run_time.sum:.3f} ({run_time.val:.3f})\t'
                             'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                             'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                             'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                    epoch, args.epochs, i, len(data_loader),
                    phase='TRAINING' if training else 'EVALUATING',
                    run_time=run_time, loss=losses, top1=top1, top5=top5))
            # wandb.log({"iter val loss": losses.avg, 'iter val acc1': top1.avg, 'iter val acc5': top5.avg})

            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            all_losses.extend(loss.detach().cpu().numpy())


        cf = confusion_matrix(all_targets, all_preds).astype(float)
        cls_cnt = cf.sum(axis=1)
        cls_hit = np.diag(cf)
        cls_acc = cls_hit / cls_cnt
        output = ('Epoch: {epoch} {flag} Results: Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Loss {loss.avg:.5f}'
                      .format(epoch=epoch, flag=training, top1=top1, top5=top5, loss=losses))
        out_cls_acc = '%s Class Accuracy: %s' % (training, (np.array2string(cls_acc, separator=',', formatter={'float_kind': lambda x: "%.3f" % x})))
        print(output)
        # print('Class Average Acc : ', out_cls_acc)
        print('args.u | ', args.u, ' myLambda | ', myLambda)
        cls_p = get_wieghts_of_each_class(args, torch.tensor(all_losses).cuda(), all_targets, args.u, myLambda)



    return losses.avg, top1.avg, top5.avg, run_time.sum, majority_losses.avg, minority_losses.avg, majority_P.avg, minority_P.avg, cls_p



def train(args, data_loader, model_new, criterion, epoch, optimizer, myLambda, cls_weights):
    model_new.train()
    return forward(args, data_loader, model_new, criterion, epoch, optimizer, cls_weights, myLambda, training=True)

def validate(args, data_loader, model_new, criterion, epoch, optimizer, myLambda, cls_weights):
    # switch to evaluate model
    model_new.eval()
    return forward(args, data_loader, model_new, criterion, epoch, optimizer, cls_weights, myLambda, training=False)





if __name__ == '__main__':
    main()


