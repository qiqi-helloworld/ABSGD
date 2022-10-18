import numpy as np
import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from numpy.testing import assert_array_almost_equal
from torchvision.datasets import ImageFolder
from PIL import Image
np.random.seed(15)

from torch.utils.data import Dataset


def build_for_cifar100(size, noise):
    """ random flip between two random classes.
    """
    assert(noise >= 0.) and (noise <= 1.)

    P = (1. - noise) * np.eye(size)
    for i in np.arange(size - 1):
        P[i, i+1] = noise

    # adjust last row
    P[size-1, 0] = noise

    assert_array_almost_equal(P.sum(axis=1), 1, 1)
    return P


def multiclass_noisify(y, P, random_state=0):
    """ Flip classes according to transition probability matrix T.
    It expects a number between 0 and the number of classes - 1.
    """

    assert P.shape[0] == P.shape[1]
    assert np.max(y) < P.shape[0]

    # row stochastic matrix
    assert_array_almost_equal(P.sum(axis=1), np.ones(P.shape[1]))
    assert (P >= 0.0).all()

    m = y.shape[0]
    new_y = y.copy()
    flipper = np.random.RandomState(random_state)

    for idx in np.arange(m):
        i = y[idx]
        # draw a vector with only an 1
        flipped = flipper.multinomial(1, P[i, :], 1)[0]
        new_y[idx] = np.where(flipped == 1)[0]

    return new_y


def other_class(n_classes, current_class):
    """
    Returns a list of class indices excluding the class indexed by class_ind
    :param nb_classes: number of classes in the task
    :param class_ind: the class index to be omitted
    :return: one random class that != class_ind
    """
    if current_class < 0 or current_class >= n_classes:
        error_str = "class_ind must be within the range (0, nb_classes - 1)"
        raise ValueError(error_str)

    other_class_list = list(range(n_classes))
    other_class_list.remove(current_class)
    other_class = np.random.choice(other_class_list)
    return other_class


class cifar10Noisy(datasets.CIFAR10):
    def __init__(self, root, train=True, transform=None, target_transform=None, download=False, nosiy_rate=0.0, asym=False):
        super(cifar10Noisy, self).__init__(root, download=download, transform=transform,
                                           target_transform=target_transform)
        self.noisy_idx = []
        if asym:
            # automobile < - truck, bird -> airplane, cat <-> dog, deer -> horse
            source_class = [9, 2, 3, 5, 4]
            target_class = [1, 0, 5, 3, 7]
            for s, t in zip(source_class, target_class):
                cls_idx = np.where(np.array(self.targets) == s)[0]
                n_noisy = int(nosiy_rate * cls_idx.shape[0])
                noisy_sample_index = np.random.choice(cls_idx, n_noisy, replace=False)
                for idx in noisy_sample_index:
                    self.targets[idx] = t
            return
        elif nosiy_rate > 0:
            n_samples = len(self.targets)
            n_noisy = int(nosiy_rate * n_samples)
            print("%d Noisy samples" % (n_noisy))
            class_index = [np.where(np.array(self.targets) == i)[0] for i in range(10)]
            class_noisy = int(n_noisy / 10)
            # noisy_idx = []
            for d in range(10):
                noisy_class_index = np.random.choice(class_index[d], class_noisy, replace=False)
                self.noisy_idx.extend(noisy_class_index)
                print("Class %d, number of noisy % d" % (d, len(noisy_class_index)))
            for i in self.noisy_idx:
                self.targets[i] = other_class(n_classes=10, current_class=self.targets[i])
            print(len(self.noisy_idx))
            print("Print noisy label generation statistics:")
            for i in range(10):
                n_noisy = np.sum(np.array(self.targets) == i)
                print("Noisy class %s, has %s samples." % (i, n_noisy))
            return

    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index

class cifar100Noisy(datasets.CIFAR100):
    def __init__(self, root, train=True, transform=None, target_transform=None, download=False, nosiy_rate=0.0, asym=False, seed=0):
        super(cifar100Noisy, self).__init__(root, download=download, transform=transform, target_transform=target_transform)

        self.noisy_idx = []
        if asym:
            """mistakes are inside the same superclass of 10 classes, e.g. 'fish'
            """
            nb_classes = 100
            P = np.eye(nb_classes)
            n = nosiy_rate
            nb_superclasses = 20
            nb_subclasses = 5

            if n > 0.0:
                for i in np.arange(nb_superclasses):
                    init, end = i * nb_subclasses, (i+1) * nb_subclasses
                    P[init:end, init:end] = build_for_cifar100(nb_subclasses, n)

                    y_train_noisy = multiclass_noisify(np.array(self.targets), P=P, random_state=seed)
                    actual_noise = (y_train_noisy != np.array(self.targets)).mean()
                assert actual_noise > 0.0
                print('Actual noise %.2f' % actual_noise)
                self.targets = y_train_noisy.tolist()
            return
        elif nosiy_rate > 0:
            n_samples = len(self.targets)
            n_noisy = int(nosiy_rate * n_samples)
            print("%d Noisy samples" % (n_noisy))
            class_index = [np.where(np.array(self.targets) == i)[0] for i in range(100)]
            class_noisy = int(n_noisy / 100)
            for d in range(100):
                noisy_class_index = np.random.choice(class_index[d], class_noisy, replace=False)
                self.noisy_idx.extend(noisy_class_index)
                print("Class %d, number of noisy % d" % (d, len(noisy_class_index)))
            for i in self.noisy_idx:
                self.targets[i] = other_class(n_classes=100, current_class=self.targets[i])
            print(len(self.noisy_idx))
            print("Print noisy label generation statistics:")
            for i in range(100):
                n_noisy = np.sum(np.array(self.targets) == i)
                print("Noisy class %s, has %s samples." % (i, n_noisy))
            return

    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index


class Clothing1MDataset(Dataset):
    def __init__(self, root, txt, transform=None):
        self.img_path = []
        self.labels = []
        self.transform = transform

        # print("LT:", txt)
        # if 'test' in txt and 'ImageNet' in txt:
        #     with open(txt) as f:
        #         for line in f:
        #             img_name = '/'.join([line.split()[0].split('/')[0], line.split()[0].split('/')[2]])
        #             self.img_path.append(os.path.join(root, img_name))
        #             self.labels.append(int(line.split()[1]))
        # else:
        txt = os.path.join(root, txt)
        with open(txt) as f:
            for line in f:
                # print(line.split('\"'))
                # print(int(line.split(',')[-1]))
                self.img_path.append(os.path.join(root, line.split('\"')[1]))
                self.labels.append(int(line.split(',')[-1]))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        path = self.img_path[index]
        label = self.labels[index]

        with open(path, 'rb') as f:
            sample = Image.open(f).convert('RGB')

        if self.transform is not None:
            sample = self.transform(sample)

        return sample, label, index






class DatasetGenerator():
    def __init__(self, batchSize=128, eval_batch_size=256, dataPath='../../datasets',
                 seed=123, numOfWorkers=4, asym=False, dataset_type='cifar10',
                 is_cifar100=False, cutout_length=16, noise_rate=0.4):
        self.seed = seed
        np.random.seed(seed)
        self.batchSize = batchSize
        self.eval_batch_size = eval_batch_size
        self.dataPath = dataPath
        self.numOfWorkers = numOfWorkers
        self.cutout_length = cutout_length
        self.noise_rate = noise_rate
        self.dataset_type = dataset_type
        self.asym = asym
        self.data_loaders = self.loadData()
        return

    def getDataLoader(self):
        return self.data_loaders

    def loadData(self):

        val_dataset = None



        if self.dataset_type == 'cifar100':
            CIFAR_MEAN = [0.5071, 0.4865, 0.4409]
            CIFAR_STD = [0.2673, 0.2564, 0.2762]

            train_transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(20),
                transforms.ToTensor(),
                transforms.Normalize(CIFAR_MEAN, CIFAR_STD)])

            test_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(CIFAR_MEAN, CIFAR_STD)])

            train_dataset = cifar100Noisy(root=self.dataPath,
                                          train=True,
                                          transform=train_transform,
                                          download=True,
                                          asym=self.asym,
                                          seed=self.seed,
                                          nosiy_rate=self.noise_rate)

            test_dataset = datasets.CIFAR100(root=self.dataPath,
                                             train=False,
                                             transform=test_transform,
                                             download=True)

        elif self.dataset_type == 'cifar10':
            CIFAR_MEAN = [0.49139968, 0.48215827, 0.44653124]
            CIFAR_STD = [0.24703233, 0.24348505, 0.26158768]

            train_transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(CIFAR_MEAN, CIFAR_STD)])

            test_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(CIFAR_MEAN, CIFAR_STD)])

            train_dataset = cifar10Noisy(root=self.dataPath, train=True,
                                         transform=train_transform, download=True,
                                         asym=self.asym, nosiy_rate=self.noise_rate)

            test_dataset = datasets.CIFAR10(root=self.dataPath, train=False,
                                            transform=test_transform, download=True)
        elif self.dataset_type == 'clothing1M':

            dat_dir = "/data/qiqi/clothing1M/"


            train_transform = transforms.Compose([
                transforms.Resize(256),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.6959, 0.6537, 0.6371), (0.3113, 0.3192, 0.3214)),
            ])
            test_transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize((0.6959, 0.6537, 0.6371), (0.3113, 0.3192, 0.3214)),
            ])

            train_dataset = Clothing1MDataset(root=dat_dir, txt = 'noisy_train_kv.txt', transform=train_transform)
            val_dataset = Clothing1MDataset(root=dat_dir, txt = 'clean_val_kv.txt', transform=test_transform)
            test_dataset = Clothing1MDataset(root=dat_dir, txt = 'clean_test_kv.txt', transform=test_transform)

        else:
            raise("Unknown Dataset")

        data_loaders = {}
        data_loaders['val_dataset'] = None
        data_loaders['train_dataset'] = DataLoader(dataset=train_dataset,
                                                   batch_size=self.batchSize,
                                                   shuffle=True,
                                                   pin_memory=True,
                                                   num_workers=self.numOfWorkers)

        data_loaders['train_dataset_peer'] = DataLoader(dataset=train_dataset,
                                                   batch_size=self.batchSize,
                                                   shuffle=True,
                                                   pin_memory=True,
                                                   num_workers=self.numOfWorkers)

        data_loaders['test_dataset'] = DataLoader(dataset=test_dataset,
                                                  batch_size=self.eval_batch_size,
                                                  shuffle=False,
                                                  pin_memory=True,
                                                  num_workers=self.numOfWorkers)

        if val_dataset is not None:
            data_loaders['val_dataset'] = DataLoader(dataset=val_dataset,
                                                  batch_size=self.eval_batch_size,
                                                  shuffle=False,
                                                  pin_memory=True,
                                                  num_workers=self.numOfWorkers)

            print("Num of test %d" % (len(val_dataset)))


        print("Num of train %d" % (len(train_dataset)))
        print("Num of test %d" % (len(test_dataset)))

        return data_loaders
