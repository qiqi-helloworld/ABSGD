__author__ = 'Qi'
# Created by on 9/17/num_minor_sample.
import random
import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np
import copy
import matplotlib.gridspec as gridspec

from scipy.interpolate import make_interp_spline, BSpline


p = None
num_minor_sample = 20
num_majority_sample = 100



def calculate_p(w,bth_x,bth_y, lbd):
    pred = np.matmul(bth_x, w)
    loss = logisticLoss(bth_y, pred)

    max_loss = np.max(loss)
    exp_loss = np.exp((loss - max_loss) / lbd)
    p = exp_loss / np.sum(exp_loss)

    lossp = np.concatenate((loss, p, bth_y), axis=1)
    lossp = sorted(lossp, key=lambda x: x[0])
    lossp = np.array(lossp)
    return lossp[:, 0], lossp[:, 1], lossp[:, 2]


def fig_lambda_p(w, bth_x, bth_y):
    lamda_beta_list = [100, 10, 1, 0.5, 0.2, 0.1]

    plt.figure()
    number = len(lamda_beta_list) + 1
    cmap = plt.get_cmap('Blues')

    colors = [cmap(i) for i in np.linspace(0, 1, number)]

    for i in range(len(lamda_beta_list)):
        lbd = lamda_beta_list[i]
        loss, p, y= calculate_p(w, bth_x, bth_y, lbd)


        plt.plot(loss, p, '--',color = colors[i+1], label = r'$\lambda = $'+ str(lamda_beta_list[i]), linewidth = 3)
        plt.scatter(loss, p, s=35, c=y, cmap= plt.cm.bwr)
        # plt.plot(loss, p, '-', color = i, label = r'$\lambda = $'+ str(lamda_beta_list[i]))
    plt.hlines(1/16, -0.05, 1.3, colors='green', linestyles='solid', linewidth=4)
    plt.title(r"Influence of $\lambda$", fontsize=15)
    plt.legend()
    plt.xlabel(r'$\ell$', fontsize = 15)
    plt.ylabel(r'Robust Weights ($\widetilde{p}$)', fontsize = 15)
    plt.savefig('lambda_change.png')
    plt.show()

def logisticLoss(y, pred):
    return np.log(1+np.exp(-y*pred))


def grad_logistic(x, y, pred, p):
    EXP = np.exp(-y * pred)
    tmp = -y*x
    if p is None:
        grad =  np.matmul(np.transpose(tmp), EXP/(1+EXP) )/len(pred)

    else:
        grad = np.matmul(np.transpose(tmp), p * EXP / (1 + EXP))

    return grad


def grad_DRO(bth_x, bth_y, pred, lamda = 10.0):
    '''
    gradients of DRO
    '''
    loss = logisticLoss(bth_y, pred)
    max_loss = np.max(loss)
    exp_loss = np.exp((loss - max_loss) / lamda)
    p = exp_loss / np.sum(exp_loss)
    return grad_logistic(bth_x, bth_y, pred, p)


def grad_Reweight(bth_x, bth_y, pred, beta):
    '''
    gradients of class balanced reweighting loss + DRO
    '''
    cls_num_list = [100, num_minor_sample]
    effective_num = 1.0 - np.power(beta, cls_num_list)
    per_cls_weights = (1.0 - beta) / np.array(effective_num)
    per_cls_weights = per_cls_weights / np.sum(per_cls_weights)
    new_y = copy.deepcopy(bth_y)
    new_y[new_y == -1] = 0
    p = per_cls_weights[new_y]
    p = p/np.sum(p)

    return grad_logistic(bth_x, bth_y, pred, p)


def grad_RE_DRO(bth_x, bth_y, pred, lamda, beta):
    '''
    gradients of class balanced reweighting loss + DRO
    :param bth_x: batch x
    :param bth_y: batch x
    :param pred: preductions
    :param lamda: temperature
    :param beta: class weights parameter
    :return:
    '''
    cls_num_list = [100, num_minor_sample]
    effective_num = 1.0 - np.power(beta, cls_num_list)
    per_cls_weights = (1.0 - beta) / np.array(effective_num)
    per_cls_weights = per_cls_weights / np.sum(per_cls_weights)

    new_y = copy.deepcopy(bth_y)
    new_y[new_y == -1] = 0
    p_RE = per_cls_weights[new_y]
    p_RE = p_RE / np.sum(p_RE)
    loss = logisticLoss(bth_y, pred)
    loss = p_RE * loss

    max_loss = np.max(loss)
    exp_loss = np.exp((loss - max_loss) / lamda)
    p = exp_loss / np.sum(exp_loss)

    return grad_logistic(bth_x, bth_y, pred, p)

def train_toyexample(w, methods, bth, beta, lamda = 0.2):

    # maxIter = len(index_ls) // bth
    # print("logistic Regression:", maxIter)
    # print(maxIter)
    loss_list = []
    for i in range(maxIter):

        if bth == 120:
            bth_x = x_train
            bth_y = y_train
            # print("Global DRO")
        else:
            bth_x = x_train[index_ls[i*bth:(i+1)*bth]]
            bth_y = y_train[index_ls[i*bth:(i+1)*bth]]

        pred = np.matmul(bth_x, w)

        loss_list.append([np.mean(logisticLoss(bth_y, pred)).tolist()])
        print("iter:", i, "loss:", np.mean(logisticLoss(bth_y, pred)).tolist())

        # print('i',  i, len(bth_x))
        if i % 100000 == 0:
           # bth_x = x_train
            # bth_y = y_train
            pred = np.matmul(bth_x, w)
            loss = logisticLoss(bth_y, pred)

            max_loss = np.max(loss)
            exp_loss = np.exp((loss - max_loss) / lbd)
            if methods == "DRO":
                p = exp_loss / np.sum(exp_loss)
            else:
                p = np.zeros_like(loss) + 1/len(bth_x)

            lossp = np.concatenate((loss, p), axis=1)
            print(lossp.shape)
            lossp = sorted(lossp, key=lambda x: x[0])
            lossp = np.array(lossp)
        #
        #     # plt.hlines(1/120, 0, 1.75, colors="gray", linestyles = 'dashdot')
        #     plt.bar(list(range(1,17)), lossp[:, 1], width=0.8, color = "red")  #, 'bo-', linewidth=4
        #     # print(loss[:, 0])
        #     # plt.title("Distribution of Samples", fontsize=15)
        #     plt.yticks(fontsize = 25)
        #     plt.xticks(fontsize = 25)
        #     #plt.xlabel("Samples", fontsize=25)
        #     #plt.ylabel("$\widetilde{p}$", fontsize=25)
        #     # plt.title(r'$\tilde{p}$ vs Sample ID', fontsize = 25)
        #     plt.ylim(-0.05, 0.62)
        #
        #     plt.savefig(methods + "_stoc_16.png")
        # #   plt.show()

        if methods == 'DRO':
            w = w - eta * grad_DRO(bth_x, bth_y, pred, lamda)
        elif methods == 'ERM':
            w = w - eta * grad_logistic(bth_x, bth_y, pred, None)
        elif methods == 'Reweight':
            w = w - eta * grad_Reweight(bth_x, bth_y, pred, beta)
        elif methods == 'RE_DRO':
            w = w - eta * grad_RE_DRO(bth_x, bth_y, pred, lamda, beta)



    return w, loss_list


def ABSGD_sample_weights_plot(loss, lamda):
    '''
    :param sample: sample losses used for calculate absgd weights
    :param lamda:  hyper parameters to calculate weights
    :return: figures of weights plots
    '''
    print('Loss : ', len(loss), 'lambda : ', lamda)
    loss = loss.reshape(-1)
    loss = np.sort(loss)
    exp_loss = np.exp(loss/lamda)
    print(exp_loss)
    p = exp_loss/np.sum(exp_loss)


    sgd_weights = np.array([1]*len(loss))/len(loss)


    plt.figure(figsize=(5,5))
    plt.bar(np.array(range(16))+0.875, p, width = 0.25,color = 'red', label='absgd')
    plt.bar(np.array(range(16))+1.125, sgd_weights, width = 0.25, color = 'blue', label='SGD')
    # np.array(range(len(loss))) + 0.5, np.array(range(len(loss)))+1,
    # plt.hist(p,  16, color = 'red', label='absgd')
    # plt.hist(sgd_weights, 16,  color = 'blue', label='SGD')
    plt.ylabel(r' $p$', fontsize = 15)
    plt.xlabel(r'Sample Index', fontsize=15)
    plt.title('Robust Weights', fontsize = 25)
    plt.legend(fontsize = 15)
    plt.savefig('./Ablation_Study/illustration_pic/robust_weights_bth_16.png')

    plt.show()




if __name__ == '__main__':

    """
    Load Data
    """
    if os.path.exists('toy_example.csv'):
        data = pd.read_csv('toy_example.csv')
        data = data/100
        minor_x = data['minor_x']
        minor_y = data['minor_y']
        majority_x = data['majority_x']
        majority_y = data['majority_y']
    else:
        mu = 60
        sigma = 5
        minor_x = []
        minor_y = []
        majority_x = []
        majority_y = []
        for i in range(100):
            if i <= num_minor_sample:
                tempX = random.gauss(mu, sigma)
                minor_x.append(tempX)
                tempY = random.gauss(mu, sigma)
                minor_y.append(tempY)
            else:
                minor_x.append(-1)
                minor_y.append(-1)

        for i in range(100):
            tempX = random.gauss(mu-num_minor_sample, sigma+5)
            majority_x.append(tempX)
            tempY = random.gauss(mu-num_minor_sample, sigma+5)
            majority_y.append(tempY)




    pos_samples = np.array(data[['minor_x', 'minor_y']])[0:num_minor_sample]
    pos_target = np.array([[1]]*num_minor_sample)
    neg_samples = np.array(data[['majority_x', 'majority_y']])
    neg_target = np.array([[-1]]*num_majority_sample)
    x_train = np.vstack((pos_samples, neg_samples))
    y_train = np.vstack((pos_target, neg_target))
    x_train = np.hstack((x_train, np.array([[1]]*(num_majority_sample+num_minor_sample))))
    # w_0 = np.array([-1, -1, 1.15])
    #w_0 = np.array([0,0,0])
    # w_0 = np.array([10.23435738, 10.83358531, -12.37028814])
    # w_0 = np.array([7.33091949, 7.16490501, -8.79802186])
    '''
    Initialization
    '''
    w_0 = np.array([5.99308785, 6.13973099, -8.03277303])
    w_0 = w_0.reshape((3,1))
    # w = copy.deepcopy(w_0)

    '''
       Hyperparameters
    '''
    maxIter = 100000
    bth_list = [16, 32, 64, 100 + num_minor_sample]
    lamda_beta_list = [0, 0, 0.9, 0.95, 0.99, 0.999, 0.01, 0.05, 0.1, 0.5, 1, 5]
    random.seed(777)
    ada_beta = [0.05, 0.2, 0.1, 0.05, 0.03]
    title =['SGD', 'Stochastic Robust Weighting']
    methods = ['ERM', 'DRO' ]
    lbd = 0.05
    beta = 0.999
    bth =128
    eta = 0.1

    index_ls = np.random.choice(range(num_majority_sample + num_minor_sample), size=maxIter*bth, replace=True, p=None)
    index_ls = [46, 107, 11, 119, 1, 32, 68, 4, 110, 83, 115, 62, 56, 9, 14, 100] + index_ls.tolist()
    indexed_samples_for_ABSGD = [46, 107, 11, 119, 1, 32, 68, 4, 110, 83, 115, 62, 56, 9, 14, 100]
    # index_ls = [23, 35, 11, 119, 1, 32, 68, 4, 35, 83, 104, 118, 56, 9, 14, 21]


    for i in range(1, len(title)):
        if i == 1:
            lbd = 0.2
        else:
            lbd = 0.1
        if i == 3:
            beta = 0.99
        w = copy.deepcopy(w_0)
        # w1 = w
        w1, loss = train_toyexample(w, methods[i], 16, beta, lbd)

        pred = np.matmul(x_train[indexed_samples_for_ABSGD], w)
        i_loss = logisticLoss(y_train[indexed_samples_for_ABSGD], pred)

        ABSGD_sample_weights_plot(i_loss, lbd)
        # print(title[i], "weights:", w1)

        # ax = fig.add_subplot(gs[i//2, i%2])
        # plt.plot(loss)
        # plt.show()

        # plt.plot(x, -w1[0] / w1[1] * x - w1[2] / w1[0], '-', color='black', linewidth=3, label='DRO')
        # plt.fill_between(x, -w1[0] / w1[1] * x - w1[2] / w1[0], interpolate=True, color='antiquewhite') # antiquewhite
        # plt.fill_between(x, -w1[0] / w1[1] * x - w1[2] / w1[0], 100, interpolate=True, color='lightcyan') # lightcyan
        #
        # plt.plot(minor_x[0:num_minor_sample], minor_y[0:num_minor_sample], 'o', color = 'yellowgreen') #salmon
        # plt.scatter(minor_x[0:num_minor_sample], minor_y[0:num_minor_sample], color = 'green', linewidth=4)
        # plt.plot(majority_x, majority_y,'o', color = 'royalblue')
        # plt.scatter(majority_x, majority_y, color='blue', linewidth=3)
        #
        # # # print(x_train[index_ls][:,0], y_train[index_ls])
        # plt.scatter(x_train[index_ls][:,0], x_train[index_ls][:,1], color = 'red', s = 100 )
        # plt.xticks([])
        # plt.yticks([])
        # plt.xlim(min(majority_x) - 0.05, max(minor_x)+0.05)
        # plt.ylim(min(majority_y) - 0.05, max(minor_y)+0.05)
        # plt.title(title[i], fontsize = 25)
        # plt.savefig(title[i] + '.png')
        # print('--i---', i)
        #plt.savefig("init" + '.png')
        # plt.show()


    # if __name__ == '__main__':x_train
    #    pass
     # fig_lambda_p(w_0, x_train[index_ls], y_train[index_ls])


