__author__ = 'Qi'
# Created by on 6/9/22.
import torch
from sklearn.metrics import auc, roc_auc_score, average_precision_score, precision_recall_curve, balanced_accuracy_score, f1_score, roc_curve
import copy, os
import numpy as np
import logging.config
from torchvision.datasets.folder import ImageFolder
import pandas as pd

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


def saved_path_res(args):
    setup_logging(os.path.join(args.save_path, 'log.txt'))
    results_file = os.path.join(args.save_path, args.res_name + '_results.%s')
    results = ResultsLog(results_file % 'csv', results_file % 'html')
    return results


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


def pred_class(output, topk=(1,)):
    with torch.no_grad():
        maxk = max(topk)
        _, pred = output.topk(maxk, 1, True, True)

    return pred.view(-1)

def binaryAccuracy(output, target):
    # print("Target:", target.size())

    batch_size = target.size(0)
    output = output.view(-1)
    target = target.view(-1)
    output[output < 0.5] = 0.0
    output[output >= 0.5] = 1.0
    correct = sum(target.eq(output))

    return correct * 1.0 / batch_size  # Accuracy


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


def auprc(targets, preds):
    precision, recall, _ = precision_recall_curve(targets, preds)
#    disp = PrecisionRecallDisplay(precision=precision, recall=recall)
    return auc(recall, precision)

def ap(targets, preds):
    return average_precision_score(targets, preds)

def ml_auroc(targets, preds):
    return roc_auc_score(targets, preds, average=None).mean()

def ml_f1_score(targets, preds):
    return f1_score(targets, preds, average=None).mean()


def balanced_accuracy(targets, preds):
    return balanced_accuracy_score(targets, preds)


def attributes_accuracy_per_sample(targets, preds):
    '''
        calculates how many attributes have been correctly predicted per sample
    '''



    dpt = copy.deepcopy(preds.detach())
    dpt[dpt >= 0] = 1
    dpt[dpt<0] = 0
    sample_acc = torch.mean((dpt == targets)*1.0, 1, keepdim=True)
    # print(sample_acc)
    return sample_acc


class MyImageFolder(ImageFolder):
    def __init__(self, root, transform):
        super().__init__(root, transform=transform)

    def __getitem__(self, index):
        sample, target = super().__getitem__(index)
        return index, sample, target


def auprc(targets, preds):
    precision, recall, _ = precision_recall_curve(targets, preds)
    #    disp = PrecisionRecallDisplay(precision=precision, recall=recall)
    return auc(recall, precision)


def compute_cla_metric(targets, preds, num_tasks):
    prc_results = []
    roc_results = []
    for i in range(num_tasks):
        is_labeled = targets[:, i] == targets[:, i]  ## filter some samples without groundtruth label
        target = targets[is_labeled, i]
        pred = preds[is_labeled, i]
        try:
            prc = auprc(target, pred)
        except ValueError:
            prc = np.nan
            print("In task #", i + 1, " , there is only one class present in the set. PRC is not defined in this case.")
        try:
            roc = roc_auc_score(target, pred)
        except ValueError:
            roc = np.nan
            print("In task #", i + 1, " , there is only one class present in the set. ROC is not defined in this case.")
        if not np.isnan(prc):
            prc_results.append(prc)
        else:
            print("PRC results do not consider task #", i + 1)
        if not np.isnan(roc):
            roc_results.append(roc)
        else:
            print("ROC results do not consider task #", i + 1)
    return prc_results, roc_results






