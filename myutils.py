
import warnings
warnings.filterwarnings(action='ignore')

# import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from torch.autograd import Variable
import logging.config
import shutil
import pandas as pd
import torch
import os
import numpy as np
import datetime
from sklearn.manifold import TSNE

# def my_warn():
#     pass
# warnings.warn = my_warn
#

def saved_path_res(args):



    if args.dataset == 'ina' or args.dataset == 'imagenet-LT' or args.dataset == 'places-LT' or args.dataset == 'covid-LT' or args.dataset == 'iNaturalist18':
        if args.save is '':
            args.save = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        if 'ldam' in args.loss_type:
            args.res_name = 'maxm_' + str(args.maxm) + '_s_' + str(args.C) + '_' + args.res_name
        if 'Reweight' in args.train_rule:
            args.res_name = 'beta_' + str(args.beta) + '_' + args.res_name
        elif 'DRW' in args.train_rule:
            args.res_name = 'beta_' + str(args.beta) + '_' + args.res_name

        if 'DRO' in args.train_rule:
            args.res_name = 'beta_' + str(args.beta) + '_' + args.res_name + '_lda_' + str(args.lamda)

        args.res_name = 'ablation_study_' + args.res_name + '_' + args.lr_schedule + '_ISPRT_' + str(args.pretrained) + '_DP_'+ str(args.DP)  \
                        + '_CB_Epoch_' + str(args.CB_shots) + '_lamda_shots_'+ str(args.lamda_shots) + '_lambda_' + str(args.lamda) + '_gamma_' + str(args.drogamma)
        save_path = os.path.join(args.results_dir, args.save, args.res_name)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        else:
            pass
    else:

        # if args.save is '':
        #     args.save = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

        args.res_name = "_".join([args.train_rule,
           args.loss_type, "lr", str(args.lr), "imr",str(args.imb_factor), args.imb_type,
              "tau", str(args.init_lamda), str(args.lamda), "bth", str(args.batch_size), "epochs", str(args.epochs), "arch", str(args.model), 'repeats',
             str(args.repeats), 'lbd_shots', str(args.lamda_shots), 'cb_shots', str(args.CB_shots), 'DP', str(args.DP)])
        args.root_log = "PAMI_TrainingResults/" + args.dataset + "/" + args.loss_type + "/" + args.res_name

        if 'rob' in args.loss_type:
            args.root_log  += '_robgamma_'+ str(args.drogamma) +'_robalpha_' + str(args.robAlpha) + '_randseed_' + str(args.seed)
        if not os.path.exists(args.root_log):
            print("Hello, Recursively Making Directories.")
            os.makedirs(args.root_log, exist_ok=True)
        if not os.path.exists(args.root_model):
            os.makedirs(args.root_model, exist_ok=True)
        save_path = args.root_log


    setup_logging(os.path.join(save_path, 'log.txt'))
    results_file = os.path.join(save_path, args.res_name + '_results.%s')
    results = ResultsLog(results_file % 'csv', results_file % 'html')
    return save_path, results

class ImbalancedDatasetSampler(torch.utils.data.sampler.Sampler):

    def __init__(self, dataset, indices=None, num_samples=None):
                
        # if indices is not provided, 
        # all elements in the dataset will be considered
        self.indices = list(range(len(dataset))) \
            if indices is None else indices
            
        # if num_samples is not provided, 
        # draw `len(indices)` samples in each iteration
        self.num_samples = len(self.indices) \
            if num_samples is None else num_samples
            
        # distribution of classes in the dataset 
        label_to_count = [0] * len(np.unique(dataset.targets))
        for idx in self.indices:
            label = self._get_label(dataset, idx)
            label_to_count[label] += 1
            
        beta = 0.9999
        effective_num = 1.0 - np.power(beta, label_to_count)
        per_cls_weights = (1.0 - beta) / np.array(effective_num)

        # weight for each sample
        weights = [per_cls_weights[self._get_label(dataset, idx)]
                   for idx in self.indices]
        self.weights = torch.DoubleTensor(weights)
        
    def _get_label(self, dataset, idx):
        return dataset.targets[idx]
                
    def __iter__(self):
        return iter(torch.multinomial(self.weights, self.num_samples, replacement=True).tolist())

    def __len__(self):
        return self.num_samples

def calc_confusion_mat(val_loader, model, args):

    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for i, (inputs, target) in enumerate(val_loader):
            if args.gpus is not None:
                target = target.cuda(args.gpu, non_blocking=True)
                input_var = Variable(inputs.type(args.type))
                target_var = Variable(target)
                output = model(input_var)
                _, pred = torch.max(output, 1)
                all_preds.extend(pred.cpu().numpy())
                all_targets.extend(target_var.cpu().numpy())

    cf = confusion_matrix(all_targets, all_preds).astype(float)
    cls_cnt = cf.sum(axis=1)
    cls_hit = np.diag(cf)
    cls_acc = cls_hit / cls_cnt
    major_covid = []
    hit_major = 0
    cnt_major = 0
    for i in range(len(args.cls_num_list)):
        if i!=3:
            hit_major += cls_hit[i]
            cnt_major += cls_cnt[i]
    major_covid.append(hit_major/cnt_major)
    major_covid.append(cls_hit[len(args.cls_num_list)-1]/cls_cnt[len(args.cls_num_list)-1])
    major_covid.append(sum(cls_hit)/sum(cls_cnt))

    # print('Class Accuracy: ')
    # print(cls_acc)
    # classes = [str(x) for x in args.cls_num_list]
    # plot_confusion_matrix(all_targets, all_preds, classes)
    # plt.savefig(os.path.join(args.root_log, args.store_name, 'confusion_matrix.png'))
    return major_covid, cls_acc

# def plot_confusion_matrix(y_true, y_pred, classes,
#                           normalize=False,
#                           title=None,
#                           cmap=plt.cm.Blues):
#     if not title:
#         if normalize:
#             title = 'Normalized confusion matrix'
#         else:
#             title = 'Confusion matrix, without normalization'
#
#     # Compute confusion matrix
#     cm = confusion_matrix(y_true, y_pred)
#
#     fig, ax = plt.subplots()
#     im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
#     ax.figure.colorbar(im, ax=ax)
#     # We want to show all ticks...
#     ax.set(xticks=np.arange(cm.shape[1]),
#            yticks=np.arange(cm.shape[0]),
#            # ... and label them with the respective list entries
#            xticklabels=classes, yticklabels=classes,
#            title=title,
#            ylabel='True label',
#            xlabel='Predicted label')
#
#     # Rotate the tick labels and set their alignment.
#     plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
#              rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax

def prepare_folders(args):
    folders_util = [args.root_log, args.root_model,
                    os.path.join(args.root_log, args.store_name),
                    os.path.join(args.root_model, args.store_name)]
    for folder in folders_util:
        if not os.path.exists(folder):
            print('creating folder ' + folder)
            os.mkdir(folder)

def save_checkpoint(args, state, is_best):
    filename = '%s/%s_ckpt.pth.tar' % (args.root_model, args.store_name)
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, filename.replace('pth.tar', 'best.pth.tar'))

def save_checkpoint_iter(state, is_best, path='.',):
    filename = os.path.join(path, '%s-th_iter_checkpoint.pth.tar' % state['iter'])
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')

def save_checkpoint_epoch(state, is_best, path='.',):
    filename = os.path.join(path, '%s-th_epoch_checkpoint.pth.tar' % state['epoch'])
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')

def save_best_checkpoint_epoch(state, is_best, path='.',):
    filename = os.path.join(path, 'model_best.pth.tar')
    torch.save(state, filename)


class AverageMeter(object):
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        if self.count == 0:
            self.count = 1
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

def COVID_accuracy(output, target):

        class_res =[]
        ave_class= 0
        for i in range(len(4)):
            class_index = target == i
            class_index = class_index.nonzero().view(-1)
            class_pred = output[class_index]
            _, class_pred = class_pred.topk(1, 1, True, True)
            cls_num = torch.sum(class_pred.view(-1) == i)
            class_res.append([torch.sum(class_pred.view(-1) == i), len(class_index)])
            ave_class+=cls_num
        ave_class = ave_class/len(target)
        return ave_class, class_res

        #pneumonia_index = target != 3
        #pneumonia_index = pneumonia_index.nonzero().view(-1)
        #pneumonia_pred = output[pneumonia_index]
        # _, covid19_pred = pneumonia_pred.topK(3, 1, True, True)

def accuracy(output, target, topk=(1,)):
    
    with torch.no_grad():

        # target = target.view(-1, 1)
        # print("target===>", target.size())
        # print("output===>", output.size())
        maxk = max(topk)
        batch_size = output.size(0)
        # print(batch_size)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        # print(correct.size())
        res = []
        for k in topk:
            # print(k, ":", correct[:k].reshape(-1).float().sum(0, keepdim=True))
            # print(k, ":", correct[:k].view(-1))
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))

        return res

def binaryAccuracy(output, target):
    # print("Target:", target.size())

    batch_size = target.size(0)
    output = output.view(-1)
    target = target.view(-1)
    output[output<0.5] = 0.0
    output[output>=0.5] = 1.0
    correct= sum(target.eq(output))

    return correct*1.0/batch_size # Accuracy

def load_checkpoint_iter(epoch, ith_init=None,  path='.'):

    #print("Ith_init:", ith_init)
    if ith_init != None:
        filename = os.path.join(path, 'init'+ str(ith_init) + '_epoch_%s-th_iter_checkpoint.pth.tar' % str(epoch))
        print(filename)
    else:
        filename = os.path.join(path, '%s-th_epoch_checkpoint.pth.tar' % str(epoch))
    #if not os.path.isfile(filename):
    #    filename = os.path.join(path, 'epoch_%s-th_iter_checkpoint.pth.tar' % str(epoch-1))
        #print(filename, "Hahaha, Horrible")
    return torch.load(filename)

def setup_logging(log_file='log.txt'):
    """Setup logging configuration
    """
    logging.basicConfig(level=logging.DEBUG,
                        format="%(asctime)s - %(levelname)s - %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S",
                        filename=log_file,
                        filemode='w')
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)

class ResultsLog(object):

    def __init__(self, path='pic_results.csv', plot_path=None):
        self.path = path
        self.plot_path = plot_path or (self.path + '.html')
        self.figures = []
        self.results = None

    def add(self, **kwargs):
        df = pd.DataFrame([kwargs.values()], columns=kwargs.keys())
        if self.results is None:
            self.results = df
        else:
            self.results = self.results.append(df, ignore_index=True)

    def save(self, title='Training Results'):
        if len(self.figures) > 0:
            if os.path.isfile(self.plot_path):
                os.remove(self.plot_path)
            self.figures = []
        self.results.to_csv(self.path, index=False, index_label=False)

def network_frozen(args, model):
    last_block_number = 0
    if args.model == "resnet152":
        last_block_number = 2
    elif args.model == 'resnet50':
        last_block_number = 2
    elif args.model == 'resnet10':
        last_block_number = 0

    last_block_pattern = 'layer4.' + str(last_block_number)

    # last_block_pattern = 'layer4.'
    if args.model == 'resnet32':
        last_block_pattern = 'layer3.4'


    total_layers = 0
    for param_name, param in model.named_parameters():  # (self.networks[key]):  # frozen the first 3 block
        total_layers +=1
        if 'fc' not in param_name and "linear" not in param_name:
            param.requires_grad = False
            if args.not_frozen_last_block:
                if last_block_pattern in param_name:
                    param.requires_grad = True

    cnt_layers = 0
    for param_name, param in model.named_parameters():
        if param.requires_grad:
            cnt_layers += 1
            # print(param_name)
    print("{0}/{1} number of trained layers".format(cnt_layers, total_layers))

def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    epoch = epoch + 1
    if args.epochs == 30:  # places-LT
        if epoch > 20:
            lr = args.lr * 0.01
        elif epoch > 10:
            lr = args.lr * 0.1
        else:
            lr = args.lr
    elif args.epochs == 90: # imagenet-LT stepLR
        if epoch > 10:
            lr = args.lr * 0.1
        else:
            lr = args.lr
    elif args.epochs == 100:  # imagenet-LT stepLR

        if epoch > 70:
            lr = args.lr * 0.0005
        elif epoch > 35:
            lr = args.lr * 0.05
        else:
            lr = args.lr

    elif args.epochs == 120:
        if epoch > 90:
            lr = args.lr * 0.1
        else:
            lr = args.lr
    elif args.epochs  == 200 or args.epochs == 300:
        if epoch <= 5:
            lr = args.lr * epoch / 5
        # elif epoch >= 180:
        #     lr = args.lr * 0.0001
        #     # if it is class balanced loss
        #     # lr = args.lr * 0.0001 * 5
        elif epoch >= 160:
            lr = args.lr * 0.01
            # if it is class balanced loss
            # lr = args.lr * 0.01 * 5
        else:
            lr = args.lr

    else:
        if epoch <= 5:
            lr = args.lr * epoch / 5
        elif epoch >= 60:
             lr = args.lr * 0.1
        else:
             lr = args.lr
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr



    return lr

def get_train_rule_hyperparameters(args, train_rule = None, epoch = 0):
    '''
    :param args: all the info needed for training strategy.
    :param train_rule: which training strategy we applied to address the data imbalance issue
    :param epoch: this epoch used to decide whether we apply the training rule for the current epoch.
    :return: the hyperparameters of the corresponding training rule.
    '''
    cls_weights = None
    train_sampler = None
    if train_rule == 'None':
        train_sampler = None
        cls_weights = None
    elif train_rule == 'resample':
        train_sampler = ImbalancedDatasetSampler(train_dataset)
        cls_weights = None

    elif 'reweight' in train_rule:
        if args.train_defer:
            idx = epoch // args.CB_shots
            betas = [args.beta] * args.epochs
            betas[0] = 0
            effective_num = 1.0 - np.power(betas[idx], args.cls_num_list)
            cls_weights = (1.0 - betas[idx]) / np.array(effective_num)
            cls_weights = cls_weights / np.sum(cls_weights) * len(args.cls_num_list)
            cls_weights = torch.FloatTensor(cls_weights).cuda()
        else:
            beta = args.beta
            effective_num = 1.0 - np.power(beta, args.cls_num_list)
            cls_weights = (1.0 - beta) / np.array(effective_num)
            cls_weights = cls_weights / np.sum(cls_weights) * len(args.cls_num_list)
            cls_weights = torch.FloatTensor(cls_weights).cuda()

    return cls_weights, train_sampler

def loaded_pretrained_models(args, model):
    print("We are loading from a pretrained ce model.")
    print(args.dataset)
    if 'cifar' in args.dataset:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location='cuda:0')
            # args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            if args.gpus is not None:
                # best_acc1 may be from a checkpoint from a different GPU
                # best_acc1 = best_acc1.to(args.gpu)
                # print(">>>>>>>>>>> : ", model.state_dict().keys(), " : <<<<<<<<<<<<")
                # fc_keys = []
                # for k, v in checkpoint['state_dict'].items():
                #     if 'fc' in k:
                #         fc_keys.append(k)
                #
                #
                # print(">>>> : ", fc_keys)
                # del  checkpoint['state_dict'][fc_keys[0]]
                # del  checkpoint['state_dict'][fc_keys[1]]
                model.module.load_state_dict(checkpoint['state_dict'], strict= False)


                print("=> loaded checkpoint '{}' (epoch {})"
                      .format(args.resume, checkpoint['epoch']))
            else:
                print("=> no checkpoint found at '{}'".format(args.resume))
    elif args.dataset == 'imagenet-LT':
        model_pack = torch.load("/home/qiuzh/qiqi/ABSGD/models/pretrianed/imagenet-LT_1st_stage_resnet50.pth.tar")
        module_pretrained_dict = model_pack['state_dict']
        model.load_state_dict(module_pretrained_dict, strict = False)
    elif args.dataset == 'iNaturalist18':
        model_pack = torch.load("./models/pretrained/resnet50_uniform_e90_IMNET.pth")
        module_pretrained_dict = model_pack['state_dict_best']['feat_model']
        module_pretrained_dict.update(model_pack['state_dict_best']['classifier'])
        best_epoch = model_pack['best_epoch']
        best_acc = model_pack['best_acc']
        epoch = model_pack['epoch']
        print("best_epoch:", best_epoch, "best_acc:", best_acc, "epoch:", epoch)
        model.load_state_dict(module_pretrained_dict, strict=False)

    elif args.dataset == 'places-LT':
        if args.stage == 3:
            '''
            Loading from the 2-th pretrained model trained myself
            '''
            model_pack = torch.load("./ce_pretrained_models/Places_LT_2nd_model_best.pth.tar")
            module_pretrained_dict = model_pack['state_dict_best']

            for k, v in list(model_pack['state_dict_best']['classifier'].items()):
                # print(k)
                if 'new_fc' in k:
                    module_pretrained_dict[str.split(k, '_')[-1]] = v.cpu()
            del module_pretrained_dict['new_fc.weight']
            del module_pretrained_dict['new_fc.bias']
            model.module.load_state_dict(module_pretrained_dict)

        elif args.stage == 2:
            '''
            Loading from the 1-th pretrained model from the decoupling paper released pretrained model: Places_LT_FB_1st_resnet152.pth.tar
            '''
            model_pack = torch.load("./ce_pretrained_models/Places_LT_FB_1st_resnet152.pth.tar")
            module_pretrained_dict = model_pack['state_dict_best']['feat_model']
            module_pretrained_dict.update(model_pack['state_dict_best']['classifier'])
            best_acc = model_pack['best_acc']
            print("We are in stage: ", args.stage, " | The best_acc of the pretrained model is :", best_acc)
            model.load_state_dict(module_pretrained_dict, strict = False)



            # if args.stage == 2:
            #     use_fc_add = False
            #     feat_dim = 2048
            # elif args.stage == 3:
            #     use_fc_add = True
            #     feat_dim = 512
def get_weights_of_majority_minority_class(args, loss, targets, u, lamda):

    exploss = torch.exp(loss/lamda)
    p = exploss/(u*len(loss))
    p = p.detach()
    majority_P = p[targets < args.num_classes//2]
    miniority_P = p[targets >= args.num_classes//2]

    return majority_P.mean(), miniority_P.mean()

def get_wieghts_of_each_class(args, loss, targets, u, lamda):

    loss = torch.tensor(loss)
    exploss = torch.exp(loss / lamda)

    p = exploss / (u * len(loss))
    p = p.detach()

    # print(">>>>>: ", p[0], targets, " :<<<<<<<")

    cls_p = []
    for tg in range(args.num_classes):
        # print(np.array(targets) == tg)
        per_cls_p = p[np.array(targets) == tg]
        # print('>>>>>>', per_cls_p, np.array(per_cls_p).mean(), '>>>>>>>')
        # print('>>>>: ', tg, len(per_cls_p), ":<<<<<")
        cls_p.append(per_cls_p.mean().item())
    # print(cls_p)
    return cls_p






def get_tsne_of_sample_feature(args, data_loader, model, split_name, epoch):

    all_feat = []
    all_targets = []
    for i, (inputs, targets) in enumerate(data_loader):
        # measure data loading time
        inputs, targets = inputs.cuda(), targets.cuda()
        outputs, feat = model(inputs)
        all_feat.extend(feat.detach().cpu().numpy().tolist())
        all_targets.extend(targets.cpu().numpy().tolist())

    print('Hello TSNE ', split_name, np.array(all_feat).shape, np.array(all_targets).shape)
    # tsne_embeded = TSNE(n_components=2).fit_transform(np.array(all_feat))
    # pd.DataFrame(tsne_embeded).to_csv(args.root_log + '/' + args.res_name + '_' + split_name + '_feat_' + str(epoch) + '_epoch.csv', header=None, index=False)
    # plt.scatter(tsne_embeded[:, 0], tsne_embeded[:, 1], s=3, c=np.array(all_targets), cmap=plt.cm.Spectral, marker='x')
    # plt.title(split_name.upper(), fontsize=15)
    # plt.xticks([])
    # plt.yticks([])
    # plt.savefig(args.root_log + '/' + args.res_name + '_' + split_name + '_tsne_' + str(epoch) + '_epoch.png')
    #
    #


