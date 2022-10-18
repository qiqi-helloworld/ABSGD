__author__ = 'Qi'
# Created by on 6/8/22.
from wilds import get_dataset
from wilds.common.data_loaders import get_train_loader, get_eval_loader
from torchvision import transforms
import os
import numpy as np
from collections import Counter
from randaugment import RandAugment, FIX_MATCH_AUGMENTATION_POOL
# dataset = get_dataset(dataset="iwildcam", download=True)




def get_iWildCam_train_test_val_DataLoader(batch_size):
    # Load the full dataset, and download it if necessary


    print('The current server is ', os.uname()[1])

    if 'amax' in os.uname()[1]:
        dataset = get_dataset(dataset="iwildcam", download=True, root_dir = '/data/')
    else:
        dataset = get_dataset(dataset="iwildcam", download=True, root_dir = '/dual_data/not_backed_up/')



    # Get the training set
    train_data = dataset.get_subset(
        "train",
        transform=transforms.Compose(
            [transforms.RandomHorizontalFlip(),
            transforms.RandomCrop((448, 448)), RandAugment(2, FIX_MATCH_AUGMENTATION_POOL),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
        ),
    )



    val_data = dataset.get_subset("val",transform=transforms.Compose(
            [transforms.CenterCrop((448, 448)), transforms.ToTensor()]
        ),
    )

    # Get the test set
    test_data = dataset.get_subset(
        "test",
        transform=transforms.Compose(
            [transforms.CenterCrop((448, 448)), transforms.ToTensor()]
        ),
    )


    id_val_data = dataset.get_subset('id_val',transform=transforms.Compose(
            [transforms.CenterCrop((448, 448)), transforms.ToTensor()]
        ))


    id_test_data = dataset.get_subset('id_test' ,transform=transforms.Compose(
            [transforms.CenterCrop((448, 448)), transforms.ToTensor()])
        )
    # Prepare the standard data loader
    train_loader = get_train_loader("standard", train_data, batch_size=batch_size, num_workers = 8)
    val_loader = get_eval_loader("standard", val_data, batch_size = batch_size, num_workers = 8)
    # Prepare the data loader
    test_loader = get_eval_loader("standard", test_data, batch_size=batch_size, num_workers = 8)
    id_val_loader = get_eval_loader("standard", id_val_data, batch_size=batch_size, num_workers = 8)
    # Prepare the data loader
    id_test_loader = get_eval_loader("standard", id_test_data, batch_size=batch_size,num_workers = 8)

    # print(">>>>:", dir(train_loader.dataset))
    # print(train_loader.dataset.y_array, np.unique(train_loader.dataset.y_array), train_loader.dataset.y_size)



    return train_loader, val_loader, test_loader, id_val_loader, id_test_loader


def get_ogbMolPCBA_train_val_DataLoader(batch_size):
    print('The current server is ', os.uname()[1])

    if 'amax' in os.uname()[1]:
        dataset = get_dataset(dataset="iwildcam", download=True, root_dir='/data/')
    else:
        dataset = get_dataset(dataset="iwildcam", download=True, root_dir='/dual_data/not_backed_up/')

    # Get the training set
    train_data = dataset.get_subset(
        "train",
        transform=transforms.Compose(
            [RandAugment(20, FIX_MATCH_AUGMENTATION_POOL), transforms.Resize((448, 448)), transforms.ToTensor()]
        ),
    )

    val_data = dataset.get_subset("val", transform=transforms.Compose(
        [transforms.Resize((448, 448)), transforms.ToTensor()]
    ),
                                  )

    # Get the test set
    test_data = dataset.get_subset(
        "test",
        transform=transforms.Compose(
            [transforms.Resize((224, 224)), transforms.ToTensor()]
        ),
    )

    # Prepare the standard data loader
    train_loader = get_train_loader("standard", train_data, batch_size=batch_size)
    val_loader = get_eval_loader("standard", val_data, batch_size=batch_size)
    # Prepare the data loader
    test_loader = get_eval_loader("standard", test_data, batch_size=batch_size)
    return train_loader, val_loader, test_loader
