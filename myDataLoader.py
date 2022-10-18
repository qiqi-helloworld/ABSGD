import os

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets
from myDatasets import IMBALANCECIFAR10, IMBALANCECIFAR100, TINYIMAGENET, get_cls_num_list, LT, INAT


RGB_statistics = {
    'iNaturalist18': {
        'mean': [0.466, 0.471, 0.380],
        'std': [0.195, 0.194, 0.192]
    },
    'ImageNet': {
        'mean': [0.485, 0.456, 0.406],
        'std': [0.229, 0.224, 0.225]
    }
}



def get_train_val_test_loader(args, train_sampler = None):
    print("==============================================>", train_sampler is None)
    if train_sampler is not None:

        sampler_dic = {
            'sampler': get_sampler(),
            'params': {'num_samples_cls': 4}
            }
    else:
        sampler_dic = None

    test_loader = None

    if args.dataset == 'ina': # useless
        args.data_root = './data/ina/images/'
        args.train_file = './data/ina/train2019.json'
        args.val_file = './data/ina/val2019.json'

        # IMG SIZE 229: INAT
        train_data = INAT(args.data_root, args.train_file, is_train=True)
        val_data = INAT(args.data_root, args.val_file, is_train=False)
        train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True,
                                               num_workers=args.works, pin_memory=True)
        val_loader = torch.utils.data.DataLoader(val_data, batch_size=args.batch_size // 6, shuffle=False,
                                             num_workers=args.works, pin_memory=True)
        args.cls_num_list = get_cls_num_list(args)

    elif args.dataset == 'imagenet-LT':
        if 'argon' in os.uname()[1]:
            args.data_root ="/nfsscratch/qqi7/imagenet/"
        elif 'amax' in os.uname()[1]: # 210.28.134.11
            args.data_root = "/data/imagenet/imagenet/"
        elif 'test-X11DPG-OT' in os.uname()[1]:
            args.data_root = "/home/qiuzh/imagenet/"
        else:
            args.data_root = '/dual_data/not_backed_up/imagenet-2012/ilsvrc/'

        train_loader =  myDataLoader_imagenet(args, args.data_root, args.batch_size, 'train', sampler_dic = sampler_dic, num_workers = args.works, shuffle = True)
        val_loader =  myDataLoader_imagenet(args, args.data_root, args.batch_size, 'val', sampler_dic = sampler_dic, num_workers = args.works, shuffle = False)
        test_loader =  myDataLoader_imagenet(args, args.data_root, args.batch_size, 'test', sampler_dic = sampler_dic, num_workers = args.works, shuffle = False)
        args.cls_num_list = get_cls_num_list(args)

    elif args.dataset == 'imagenet':
        if 'argon' in os.uname()[1]:
            args.data_root ="/nfsscratch/qqi7/imagenet/"
        elif 'amax' in os.uname()[1]: # 210.28.134.11
            args.data_root = "/data/imagenet/imagenet/"
        elif 'test-X11DPG-OT' in os.uname()[1]:
            args.data_root = "/home/qiuzh/imagenet/"
        else:
            args.data_root = '/dual_data/not_backed_up/imagenet-2012/ilsvrc/'

        train_loader = myDataLoader_imagenet(args, args.data_root, args.batch_size, 'train', sampler_dic=sampler_dic,
                                          num_workers=args.works, shuffle=True)
        val_loader = myDataLoader_imagenet(args, args.data_root, args.batch_size // 4, 'val', sampler_dic=sampler_dic,
                                        num_workers=args.works, shuffle=False)
        test_loader = myDataLoader_imagenet(args, args.data_root, args.batch_size // 4, 'test', sampler_dic=sampler_dic,
                                         num_workers=args.works, shuffle=False)
        args.cls_num_list = get_cls_num_list(args)

    elif args.dataset == 'places-LT':
        if 'argon' in os.uname()[1]:
            args.data_root = "/Users/qqi7/places/"
        elif 'amax' in os.uname()[1]:
            args.data_root = "/data/qiqi/Places_LT/"
        else:
            args.data_root = "/dual_data/not_backed_up/places/"


        train_loader = myDataLoader_Places_LT(args, args.data_root, args.batch_size, 'train', sampler_dic=sampler_dic,
                                          num_workers=args.works, shuffle=True)
        val_loader = myDataLoader_Places_LT(args, args.data_root, args.batch_size, 'val', sampler_dic=sampler_dic,
                                        num_workers=args.works, shuffle=False)
        test_loader = myDataLoader_Places_LT(args, args.data_root, args.batch_size, 'test', sampler_dic=sampler_dic,
                                         num_workers=args.works, shuffle=False)
        args.cls_num_list = get_cls_num_list(args)

    elif args.dataset == 'iNaturalist18':
        if 'argon' in os.uname()[1]:
            args.data_root = "/nfsscratch/qqi7/iNaturalist2018/"
        elif 'amax' in os.uname()[1]: # 210.28.134.11
            args.data_root = "/data/iNaturalist2018/"
        else:
            args.data_root = "/dual_data/not_backed_up/iNaturalist2018/"

        train_loader = myDataLoader_iNaturalist18(args, args.data_root, args.batch_size, 'train', sampler_dic=None,
                                           num_workers=args.works, shuffle=True)
        val_loader = myDataLoader_iNaturalist18(args, args.data_root, args.batch_size // 4, 'val', sampler_dic=None,
                                         num_workers=args.works, shuffle=False)
        args.cls_num_list = get_cls_num_list(args)

    elif args.dataset == 'covid-LT':

        if 'argon' in os.uname()[1]:
            args.data_root = "/nfsscratch/qqi7/COVID-19/"
        else:
            args.data_root = "/dual_data/not_backed_up/CheXpert_COVID/"

        train_loader = myDataLoader_Covid_LT(args, args.data_root, args.batch_size, 'train', sampler_dic=None,
                                          num_workers=args.works, shuffle=True, imb_factor=args.imb_factor)
        val_loader = myDataLoader_Covid_LT(args, args.data_root, args.batch_size//4, 'val', sampler_dic=None,
                                          num_workers=args.works, shuffle=False)
        args.cls_num_list = get_cls_num_list(args)
    else:

        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        transform_val = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        if 'amax' in os.uname()[1]:
            cifar_root = "/data/qiuzh/cv_datasets"
        else:
            cifar_root = "./data/"


        if args.dataset == 'cifar10':
            train_dataset = IMBALANCECIFAR10(root = cifar_root, imb_type=args.imb_type, imb_factor=args.imb_factor,
                                             rand_number=args.rand_number, train=True, download=True,
                                             transform=transform_train)
            val_dataset = datasets.CIFAR10(root = cifar_root, train=False, download=True, transform=transform_val)
        elif args.dataset == 'cifar100':
            train_dataset = IMBALANCECIFAR100(root = cifar_root, imb_type=args.imb_type, imb_factor=args.imb_factor,
                                              rand_number=args.rand_number, train=True, download=True,
                                              transform=transform_train)
            val_dataset = datasets.CIFAR100(root = cifar_root, train=False, download=True, transform=transform_val)
        elif args.dataset == 'tiny-imagenet':
            transform_train = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(64, padding=8),
                transforms.ToTensor()
            ])
            transform_val = transforms.Compose([
                transforms.ToTensor()
            ])

            train_dataset = TINYIMAGENET(root='./data/tiny-imagenet/train', imb_type=args.imb_type,
                                         imb_factor=args.imb_factor, rand_number=args.rand_number,
                                         transform=transform_train)
            val_dataset = datasets.ImageFolder(root='./data/tiny-imagenet/val', transform=transform_val)
        else:
            warnings.warn('Dataset is not listed')
            return

        cls_num_list = train_dataset.get_cls_num_list()
        args.cls_num_list = cls_num_list

        #if train_sampler != 'None':
        #   train_sampler = ClassAwareSampler
        # else:

        train_sampler = None

        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
            num_workers=args.workers, pin_memory=True, sampler=None if train_sampler is None else train_sampler(train_dataset, num_samples_cls=4))

        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True)
        print('len data loader', len(train_loader), len(val_loader))

    return train_loader, val_loader, test_loader

#/dual_data/not_backed_up/imagenet-2012/ilsvrc
def myDataLoader_imagenet(args, data_root, batch_size, phase, sampler_dic=None, num_workers=4, shuffle=True):
    assert phase in {'train', 'val', 'test'}
    if 'LT' in args.dataset:
        key = 'ImageNet_LT'
        txt = f'./data/ImageNet_LT/ImageNet_LT_{phase}.txt'
    else:
        key = 'ImageNet'
        txt = f'./data/ImageNet/ImageNet_{phase}.txt'

    rgb_mean, rgb_std = RGB_statistics['ImageNet']['mean'], RGB_statistics['ImageNet']['std']


    if phase == 'val' and args.stage == 2:
        transform = get_data_transform('train', rgb_mean, rgb_std)
    else:
        transform = get_data_transform(phase, rgb_mean, rgb_std)

    set_imagenet = LT(data_root, txt, transform)
    print(f'===> {phase} data length {len(set_imagenet)}')

    # if phase == 'test' and test_open:
    #     open_txt = './data/%s/%s_open.txt' % (dataset, dataset)
    #     print('Testing with open sets from %s' % open_txt)
    #     open_set_ = INaturalist('./data/%s/%s_open' % (dataset, dataset), open_txt, transform)
    #     set_ = ConcatDataset([set_, open_set_])

    if sampler_dic and phase == 'train':
        print('Using sampler: ', sampler_dic['sampler'])
        print('Sampler parameters: ', sampler_dic['params'])
        return DataLoader(dataset=set_imagenet, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                          sampler=sampler_dic['sampler'](set_imagenet, **sampler_dic['params']))
    else:
        print('No sampler.')
        print('Shuffle is %s.' % shuffle)
        return DataLoader(dataset=set_imagenet, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

def myDataLoader_Places_LT(args, data_root, batch_size, phase, sampler_dic=None, num_workers=4, shuffle=True):
    assert phase in {'train', 'val', 'test'}
    key = 'ImageNet'
    txt = f'./data/Places_LT/Places_LT_{phase}.txt'
    # print(f'===> Loading Places data from {txt}')
    rgb_mean, rgb_std = RGB_statistics[key]['mean'], RGB_statistics[key]['std']

    if phase == 'val' and args.stage == 2:
        transform = get_data_transform('train', rgb_mean, rgb_std)
    else:
        transform = get_data_transform(phase, rgb_mean, rgb_std)

    set_imagenet = LT(data_root, txt, transform)
    # print(f'===> {phase} data length {len(set_imagenet)}')

    # if phase == 'test' and test_open:
    #     open_txt = './data/%s/%s_open.txt' % (dataset, dataset)
    #     print('Testing with open sets from %s' % open_txt)
    #     open_set_ = INaturalist('./data/%s/%s_open' % (dataset, dataset), open_txt, transform)
    #     set_ = ConcatDataset([set_, open_set_])

    if sampler_dic and phase == 'train':
        print('Using sampler: ', sampler_dic['sampler'])
        print('Sampler parameters: ', sampler_dic['params'])
        return DataLoader(dataset=set_imagenet, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                          sampler=sampler_dic['sampler'](set_imagenet, **sampler_dic['params']))
    else:
        print('No sampler.')
        print('Shuffle is %s.' % shuffle)
        return DataLoader(dataset=set_imagenet, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

def myDataLoader_iNaturalist18(args, data_root, batch_size, phase, sampler_dic=None, num_workers=4, shuffle=True, imb_factor = 0.01):
    assert  phase in {'train', 'val'} , "There is no test phase for iNaturalist18"
    key = 'iNaturalist18'
    txt = f'./data/iNaturalist18/iNaturalist18_{phase}.txt'

    print(f'===> Loading iNaturalist10 data from {txt}')
    rgb_mean, rgb_std = RGB_statistics[key]['mean'], RGB_statistics[key]['std']


    if phase == 'val' and args.stage == 2:
        transform = get_data_transform('train', rgb_mean, rgb_std)
    else:
        transform = get_data_transform(phase, rgb_mean, rgb_std)


    set_imagenet = LT(data_root, txt, transform)
    print(f'===> {phase} data length {len(set_imagenet)}')

    if sampler_dic and phase == 'train':
        print('Using sampler: ', sampler_dic['sampler'])
        print('Sampler parameters: ', sampler_dic['params'])
        return DataLoader(dataset=set_imagenet, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                          sampler=sampler_dic['sampler'](set_imagenet, **sampler_dic['params']))
    else:
        print('No sampler.')
        print('Shuffle is %s.' % shuffle)
        return DataLoader(dataset=set_imagenet, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)


def myDataLoader_Covid_LT(args, data_root, batch_size, phase, sampler_dic=None, num_workers=4, shuffle=True, imb_factor = 0.01):

    assert  phase in {'train', 'val'} , "There is no test phase for Covid_LT"
    key = 'ImageNet'
    if phase == 'train':
        txt = f'./data/Covid_LT/{str(imb_factor)}_Covid_LT_{phase}.txt'
    else:
        txt = f'./data/Covid_LT/Covid_LT_{phase}.txt'
    print(f'===> Loading Places data from {txt}')
    rgb_mean, rgb_std = RGB_statistics[key]['mean'], RGB_statistics[key]['std']

    if phase == 'val' and args.stage== 2:
        transform = get_data_transform('train', rgb_mean, rgb_std)
    else:
        transform = get_data_transform(phase, rgb_mean, rgb_std)

    set_imagenet = LT(data_root, txt, transform)
    print(f'===> {phase} data length {len(set_imagenet)}')

    if sampler_dic and phase == 'train':
        print('Using sampler: ', sampler_dic['sampler'])
        print('Sampler parameters: ', sampler_dic['params'])
        return DataLoader(dataset=set_imagenet, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                          sampler=sampler_dic['sampler'](set_imagenet, **sampler_dic['params']))
    else:
        print('No sampler.')
        print('Shuffle is %s.' % shuffle)
        return DataLoader(dataset=set_imagenet, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

def get_data_transform(split, rgb_mean, rbg_std, key='ImageNet'):

        data_transforms = {
            'train': transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(rgb_mean, rbg_std)
            ]) if key == 'iNaturalist18' else transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0),
                transforms.ToTensor(),
                transforms.Normalize(rgb_mean, rbg_std)
            ]),
            'val': transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(rgb_mean, rbg_std)
            ]),
            'test': transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(rgb_mean, rbg_std)
            ])
        }
        return data_transforms[split]

