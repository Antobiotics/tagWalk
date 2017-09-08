import numpy as np
import pandas as pd

import torch

from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader

from sklearn.preprocessing import MultiLabelBinarizer
import torchvision.transforms as transforms

from PIL import Image

import fachung.configuration as conf

BASE_PATH = (
    conf.BASE_DATA +
    conf.get_config()
    .get(conf.MODE, 'tag_walk')
)

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

DEFAULT_TRANSFORMS = (
    transforms.Compose([
        transforms.RandomSizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ])
)


class TagwalkDataset(Dataset):
    def __init__(self, csv_path=None, img_path=None, transform=None):
        if csv_path is None:
            csv_path = BASE_PATH + '/assocs.csv'

        if img_path is None:
            img_path = BASE_PATH + 'images/all'

        self.reference_dataset = self.read_reference_dataset(csv_path)

        self.mlb = MultiLabelBinarizer()
        self.img_path = img_path
        self.transform = transform

        self.X_train = self.reference_dataset['image']
        self.y_train = self.mlb.fit_transform(
            self.reference_dataset['tags']
        ).astype(np.float32)

        self.num_classes = self.y_train[0].shape[0]

    def read_reference_dataset(self, csv_path):
        tmp_df = (
            pd.read_csv(csv_path)
            .groupby('image')['tag']
            .apply(list)
        ).reset_index()
        tmp_df.columns = ['image', 'tags']
        # return tmp_df
        return tmp_df.head(n=300)

    def get_image(self, index):
        item_img_path = '/'.join([
            self.img_path,
            self.X_train[index]
        ])
        img = Image.open(item_img_path)
        img = img.convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img, item_img_path

    def get_labels(self, index):
        return torch.from_numpy(self.y_train[index])

    def __getitem__(self, index):
        img, item_img_path = self.get_image(index)
        labels = self.get_labels(index)
        return img, labels, item_img_path

    def __len__(self):
        return len(self.X_train.index)


class TagwalkSequenceDataset(TagwalkDataset):
    def get_labels(self, index):
        tw_one_hot = self.y_train[index]
        tag_idx = np.where(tw_one_hot == 1)[0]
        return tag_idx

    def __getitem__(self, index):
        img, _ = self.get_image(index)
        labels = self.get_labels(index)
        return img, labels


def collate_sequence_data(data):
    """ Ensure all sequences have same length in the current batch
    """
    # Order batch by decreasing length
    data.sort(key=lambda x: len(x[1]), reverse=True)
    images, labels = zip(*data)

    images = torch.stack(images, 0)
    lengths = [len(label) for label in labels]
    targets = torch.zeros(len(labels), max(lengths)).long()

    for i, target in enumerate(targets):
        end = lengths[i]
        targets[i, :end] = target[:end]
    return images, targets, lengths


def tagwalk_dataloader(dataset=None):
    if dataset is None:
        dataset = TagwalkDataset()
    train_loader = DataLoader(dataset,
                              batch_size=256,
                              shuffle=True,
                              num_workers=4)
    return train_loader
