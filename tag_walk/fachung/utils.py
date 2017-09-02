import numpy as np

from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

from sklearn.model_selection import train_test_split


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
                                 pin_memory=False):

    # load the dataset
    train_dataset = dataset
    valid_dataset = dataset
    test_dataset = dataset
    # test_dataset.transform = None

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
                              pin_memory=pin_memory)

    valid_loader = DataLoader(valid_dataset,
                              batch_size=batch_size,
                              sampler=valid_sampler,
                              num_workers=num_workers,
                              pin_memory=pin_memory)

    test_loader = DataLoader(test_dataset,
                             batch_size=batch_size,
                             sampler=test_sampler,
                             num_workers=num_workers,
                             pin_memory=pin_memory)

    return (train_loader, valid_loader, test_loader)
