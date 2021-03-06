import numpy as np

import torch
import torch.autograd as autograd

from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from torch.utils.data.sampler import SubsetRandomSampler

import fachung.transforms as transforms
from sklearn.model_selection import train_test_split

USE_CUDA = torch.cuda.is_available()


def Variable(data, *args, **kwargs):
    var = autograd.Variable(data, *args, **kwargs)
    if USE_CUDA:
        var = var.cuda()
    return var

def from_numpy(ndarray):
    tensor = torch.from_numpy(ndarray).float()
    if USE_CUDA:
        tensor = tensor.cuda()
    return tensor

def to_tensor(array):
    tensor = torch.Tensor(array).float()
    if USE_CUDA:
        tensor = tensor.cuda()
    return tensor

def split_dataset(inputs, labels):
    test_size = inputs.shape[0] // 3
    validation_size = test_size // 2
    X_train, X_test, y_train, y_test = (
        train_test_split(inputs, labels,
                         test_size=test_size,
                         random_state=42)
    )

    X_val, X_test, y_val, y_test = (
        train_test_split(X_test, y_test,
                         test_size=validation_size,
                         random_state=42)
    )

    print('Dataset sizes:')
    print('    Train:      {}'.format(X_train.shape))
    print('    Validation: {}'.format(X_val.shape))
    print('    Test:       {}'.format(X_test.shape))

    return X_train, X_val, X_test, y_train, y_val, y_test


def get_train_valid_test_loaders(dataset, batch_size,
                                 random_seed,
                                 valid_size=0.2,
                                 shuffle=True,
                                 num_workers=4,
                                 pin_memory=False,
                                 collate_fn=default_collate):

    # load the dataset
    train_dataset = dataset
    valid_dataset = dataset
    test_dataset = dataset
    test_dataset.transform = transforms.TEST_TRANSFORMS

    num_train = len(train_dataset)
    indices = list(range(num_train))

    if shuffle:
        np.random.seed(random_seed)
        np.random.shuffle(indices)

    train_idx, val_idx, test_idx, _, _, _ = split_dataset(np.array(indices),
                                                          np.zeros(num_train))

    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(val_idx)
    test_sampler = SubsetRandomSampler(test_idx)

    train_loader = DataLoader(train_dataset,
                              batch_size=batch_size,
                              sampler=train_sampler,
                              num_workers=num_workers,
                              pin_memory=pin_memory,
                              collate_fn=collate_fn)

    valid_loader = DataLoader(valid_dataset,
                              batch_size=batch_size,
                              sampler=valid_sampler,
                              num_workers=num_workers,
                              pin_memory=pin_memory,
                              collate_fn=collate_fn)

    test_loader = DataLoader(test_dataset,
                             batch_size=batch_size,
                             sampler=test_sampler,
                             num_workers=num_workers,
                             pin_memory=pin_memory,
                             collate_fn=collate_fn)

    return (train_loader, valid_loader, test_loader)
