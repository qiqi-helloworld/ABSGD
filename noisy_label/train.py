import argparse
import torch
import time
import logging
import os
from model import SCEModel, ResNet18, ResNet34
from noisyDataset import DatasetGenerator, cifar100Noisy, cifar10Noisy
from tqdm import tqdm
from myutils.utils import AverageMeter, accuracy, count_parameters_in_MB
from train_util import TrainUtil, get_wieghts_of_clean_noisy
from loss import SCELoss, ABLoss,  f_divergence, TruncatedLoss
import torchvision.models as models
import torch.nn as nn
import wandb
import pandas as pd
# ArgParse
parser = argparse.ArgumentParser(description='SCE Loss')
parser.add_argument('--lr', type=float, default=0.1)
parser.add_argument('--l2_reg', type=float, default=5e-4)
parser.add_argument('--grad_bound', type=float, default=5) # 5
parser.add_argument('--train_log_every', type=int, default=100)
parser.add_argument('--resume', action='store_true', default=False)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--data_path', default='/data/qiuzh/cv_datasets', type=str)
parser.add_argument('--checkpoint_path', default='checkpoints', type=str)
parser.add_argument('--data_nums_workers', type=int, default=8)
parser.add_argument('--epoch', type=int, default=120)
parser.add_argument('--nr', type=float, default=0.4, help='noise_rate')
parser.add_argument('--loss', type=str, default='SCE', help='SCE, CE, ROBSCE')
parser.add_argument('--alpha', type=float, default=1.0, help='alpha scale of ce')
parser.add_argument('--beta', type=float, default=1.0, help='beta scale of asymetric ce')
parser.add_argument('--version', type=str, default='SCE0.0', help='Version')
parser.add_argument('--dataset_type', choices=['cifar10', 'cifar100', 'clothing1M'], type=str, default='cifar10')
parser.add_argument('--asym', default=False, type=bool)
parser.add_argument('--seed', type=int, default=123)
parser.add_argument('--abAlpha', type=float, default=0.5, help='robAlpha scale')
parser.add_argument('--lamda', type = float, default= 1 )
parser.add_argument('--init_lamda', type = float, default=200)
parser.add_argument('--droGamma', type = float, default=0.1, help = 'moving average parameter of gamma in dro')
parser.add_argument('--noisy_index', type = list, default=None, help = 'the index of noisy data')
parser.add_argument('--class_tau', type = float, default=0, help = 'the weights of outputs' )
parser.add_argument('--u', type = float, default=0, help = 'history moving average value' )
parser.add_argument('--lamda_shots', type = int, default=80, help = 'starting epoch of using ABSGD')



args = parser.parse_args()
GLOBAL_STEP, EVAL_STEP, TEST_BEST_ACC, TEST_BEST_ACC_TOP5 = 0, 0, 0, 0
cell_arc = None

def adjust_weight_decay(model, l2_value):
    conv, fc = [], []
    for name, param in model.named_parameters():
        # print(name)
        if not param.requires_grad:
            # frozen weights
            continue
        if 'module.fc1' in name:
            fc.append(param)
        else:
            conv.append(param)
    params = [{'params': conv, 'weight_decay': l2_value}, {'params': fc, 'weight_decay': 0.01}]
    # print(fc)
    return params



if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    device = torch.device('cuda')
else:
    device = torch.device('cpu')



def model_test(args, epoch, fixed_cnn, data_loader):
    fixed_cnn.eval()
    valid_loss_meters = AverageMeter()
    valid_acc_meters = AverageMeter()
    valid_acc5_meters = AverageMeter()
    ce_loss = torch.nn.CrossEntropyLoss(reduction='none')
    clsp = None
    all_losses = []
    all_targets = []
    for i, (images, labels, indexes) in enumerate(data_loader):
        start = time.time()
        images, labels = images.to(device), labels.to(device)
        with torch.no_grad():
            pred = fixed_cnn(images)
            loss = ce_loss(pred, labels)
            acc, acc5 = accuracy(pred, labels, topk=(1, 5))
        all_losses.extend(loss.cpu().numpy())
        all_targets.extend(labels.cpu().numpy())

        valid_loss_meters.update(loss.mean().item())
        valid_acc_meters.update(acc.item())
        valid_acc5_meters.update(acc5.item())
        end = time.time()

    clsp = None
    return valid_acc_meters.avg, valid_acc5_meters.avg, clsp


def model_eval(args, epoch, fixed_cnn, data_loader):
    fixed_cnn.eval()
    valid_loss_meters = AverageMeter()
    valid_acc_meters = AverageMeter()
    valid_acc5_meters = AverageMeter()
    ce_loss = torch.nn.CrossEntropyLoss(reduction='none')
    clsp = None
    all_losses = []
    all_targets = []
    for i, (images, labels, indexes) in enumerate(data_loader):
        start = time.time()
        images, labels = images.to(device), labels.to(device)
        with torch.no_grad():
            pred = fixed_cnn(images)
            loss = ce_loss(pred, labels)
            acc, acc5 = accuracy(pred, labels, topk=(1, 5))
        all_losses.extend(loss.cpu().numpy())
        all_targets.extend(labels.cpu().numpy())

        valid_loss_meters.update(loss.mean().item())
        valid_acc_meters.update(acc.item())
        valid_acc5_meters.update(acc5.item())
        end = time.time()
    if args.noisy_index is not None:
         norm_u = args.u if type(args.u) is int else args.u.item()

         norm_u =1
         # print(all_targets, all_losses, norm_u)
         clsp = get_wieghts_of_clean_noisy(args, all_losses, all_targets, norm_u,  args.lamda, args.noisy_index)
    print(clsp)

    return valid_acc_meters.avg, valid_acc5_meters.avg, clsp, valid_loss_meters.avg




def train_fixed(args, starting_epoch, epoch, data_loader, fixed_cnn, criterion, fixed_cnn_optmizer, fixed_cnn_scheduler):
    fixed_cnn.train()
    train_loss_meters = AverageMeter()
    train_acc_meters = AverageMeter()
    train_acc5_meters = AverageMeter()

    myLambda = args.init_lamda

    if args.dataset_type == 'clothing1M':
        if epoch >= 5:
            myLambda = args.lamda
    else:
        if epoch >= args.lamda_shots:
            myLambda = args.lamda



    peer_iter = iter(data_loader['train_dataset_peer'])
    # cnt = 0

    if args.loss == 'TCE' and epoch > args.lamda_shots:

        for batch_idx, (inputs, labels, indexes) in enumerate(data_loader["train_dataset"]):
            inputs, labels = inputs.cuda(non_blocking=True), labels.cuda(non_blocking=True)
            outputs = fixed_cnn(inputs)
            criterion.update_weight(outputs, labels, indexes)

    for i, (images, labels, indexes) in enumerate(data_loader["train_dataset"]):
        if i%500 == 0:
            print(i,'/', len(data_loader["train_dataset"]))

        images, labels = images.cuda(non_blocking = True), labels.cuda(non_blocking = True)
        pred = fixed_cnn(images)
        pred = pred / (torch.norm(pred, p=2, dim=1, keepdim=True) ** args.class_tau)

        if 'AB' in args.loss:
            loss = criterion(pred, labels, indexes, myLambda)
            args.u = criterion.u
        elif args.loss == 'f_dvg':
            input1 = peer_iter.next()[0]
            pred1 = fixed_cnn(input1)
            loss = criterion(pred, pred1, labels)
        elif args.loss == 'TCE':
            loss = criterion(pred, labels, indexes)
            loss = torch.mean(loss)
        else:
            loss = criterion(pred, labels)

        fixed_cnn_optmizer.zero_grad()
        loss.backward()
        # if args.loss == 'SCE':
        torch.nn.utils.clip_grad_norm_(fixed_cnn.parameters(), args.grad_bound)
        fixed_cnn_optmizer.step()

        # print(epoch, i, " : Stuck Point 2")
        acc, acc5 = accuracy(pred, labels, topk=(1, 5))
        acc_sum = torch.sum((torch.max(pred, 1)[1] == labels).type(torch.float))
        total = pred.shape[0]
        acc = acc_sum / total
        # print(epoch, i, " : Stuck Point 3")
        train_loss_meters.update(loss.item())
        train_acc_meters.update(acc.item())
        train_acc5_meters.update(acc5.item())

        end = time.time()


    return

def main():
    global GLOBAL_STEP, reduction_arc, cell_arc
    # Dataset
    dataset = DatasetGenerator(batchSize=args.batch_size,
                               dataPath=args.data_path,
                               numOfWorkers=args.data_nums_workers,
                               noise_rate=args.nr,
                               asym=args.asym,
                               seed=args.seed,
                               dataset_type=args.dataset_type)
    dataLoader = dataset.getDataLoader()

    wandb.init(config=args, project="noisy_labels", entity="qiqi-helloworld")

    if args.dataset_type == 'cifar100':
        num_classes = 100
        args.epoch = 120
        fixed_cnn = ResNet34(num_classes=num_classes)
        args.noisy_index = dataLoader['train_dataset'].dataset.noisy_idx
        print('>>>>> : ', args.nr, ':',  len(args.noisy_index))

    elif args.dataset_type == 'cifar10':
        num_classes = 10
        args.epoch = 120
        fixed_cnn = SCEModel()
        args.noisy_index = dataLoader['train_dataset'].dataset.noisy_idx
        # print('>>>>> : ', args.nr, ':',  len(args.noisy_index))
        #    cifar10Nosiy.noisy_idx
    elif args.dataset_type == 'clothing1M':
        num_classes = 14
        fixed_cnn = models.resnet50(pretrained=True)
        fixed_cnn.fc = nn.Linear(2048, 14)
    else:
        raise('Unimplemented')

    # print("args.loss: >>>>>> ", args.loss)
    if 'AB' in args.loss:
        # print(">>>>>> ", num_classes)
        criterion =ABLoss(loss= args.loss, alpha=args.alpha, beta=args.beta, num_classes=num_classes, abAlpha = args.abAlpha, droGamma = args.droGamma, trainset_size= len(dataLoader['train_dataset'].dataset))
    elif args.loss == 'SCE':
        criterion = SCELoss(alpha=args.alpha, beta=args.beta, num_classes=num_classes)
    elif args.loss == 'CE':
        criterion = torch.nn.CrossEntropyLoss()
    elif args.loss == 'f_divergence':
        criterion = f_divergence(num_classes = num_classes)
    elif args.loss == 'TCE':
        criterion = TruncatedLoss(trainset_size=len(dataLoader['train_dataset'].dataset))
    else:
        pass
    # print(args)
    fixed_cnn = torch.nn.DataParallel(fixed_cnn)
    fixed_cnn.to(device)
    fixed_cnn_optmizer = torch.optim.SGD(params= adjust_weight_decay(fixed_cnn, args.l2_reg),
                                         lr=args.lr,
                                         momentum=0.9)

    if args.dataset_type != 'clothing1M':
        # CIFAR10
        # fixed_cnn_scheduler = torch.optim.lr_scheduler.MultiStepLR(fixed_cnn_optmizer, milestones=[40, 80], gamma=0.1) # cifar10 40, 80
        fixed_cnn_scheduler = torch.optim.lr_scheduler.MultiStepLR(fixed_cnn_optmizer, milestones=[40, 80], gamma=0.1) # cifar10 40, 80
    else:
        fixed_cnn_scheduler = torch.optim.lr_scheduler.MultiStepLR(fixed_cnn_optmizer, milestones=[5], gamma=0.1)


    starting_epoch = 0
    global GLOBAL_STEP, reduction_arc, cell_arc, TEST_BEST_ACC, EVAL_STEP, TEST_BEST_ACC_TOP5
    print(">>>> ", TEST_BEST_ACC, TEST_BEST_ACC_TOP5)
    start_time = time.time()
    train_cur_acc, eval_curr_acc, train_cur_acc5, eval_curr_acc5, best_epoch, train_loss, eval_loss = 0, 0, 0, 0, 0, 0, 0
    test_curr_acc, test_curr_acc5, cur_test_clsp = model_test(args, 0, fixed_cnn, dataLoader['test_dataset'])
    for epoch in range(starting_epoch, args.epoch):

        print('Epoch : ', epoch)
        train_fixed(args, starting_epoch, epoch, dataLoader, fixed_cnn, criterion, fixed_cnn_optmizer, fixed_cnn_scheduler)

        fixed_cnn_scheduler.step()

        print(dataLoader['val_dataset'] is not None, dataLoader['val_dataset'] )

        if dataLoader['val_dataset'] is not None:
            eval_curr_acc, eval_curr_acc5, cur_eval_clsp, eval_loss = model_eval(args, epoch, fixed_cnn, dataLoader['val_dataset'])
        else:
            train_cur_acc, train_cur_acc5, cur_train_clsp, train_loss = model_eval(args, epoch, fixed_cnn, dataLoader['train_dataset'])

        test_curr_acc, test_curr_acc5, cur_test_clsp = model_test(args, epoch, fixed_cnn, dataLoader['test_dataset'])


        TEST_BEST_ACC = max(test_curr_acc, TEST_BEST_ACC)
        TEST_BEST_ACC_TOP5 = max(test_curr_acc5,TEST_BEST_ACC_TOP5)
        wandb.log({'test_acc' : test_curr_acc, 'best_test_acc' : TEST_BEST_ACC, 'train_loss': train_loss, 'eval_loss':eval_loss}, step = epoch)


        if test_curr_acc == TEST_BEST_ACC:
            best_epoch = epoch

        print(epoch, " : >>>>> test_curr_acc: ", args.nr, test_curr_acc, TEST_BEST_ACC, best_epoch, 'train_loss : ', train_loss, 'eval_loss :', eval_loss)



    print(epoch, TEST_BEST_ACC, 'lambda:', args.lamda, best_epoch, (time.time() - start_time)//60, 'minutes' )

if __name__ == '__main__':
    main()
