import numpy as np
import pandas as pd

import torch

from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader

from sklearn.preprocessing import MultiLabelBinarizer

from PIL import Image

import fachung.configuration as conf
from fachung.utils import from_numpy


BASE_PATH = (
    conf.BASE_DATA +
    conf.get_config()
    .get(conf.MODE, 'tag_walk')
)


class TagwalkDataset(Dataset):
    def __init__(self, csv_path=None, img_path=None, transform=None):
        if csv_path is None:
            csv_path = BASE_PATH + '/tagwalk_ref_df.csv'

        if img_path is None:
            img_path = BASE_PATH + 'images/v2/__all'

        self.reference_dataset = self.read_reference_dataset(csv_path)

        self.mlb = MultiLabelBinarizer()
        self.img_path = img_path
        self.transform = transform

        self.X_train = self.reference_dataset['image']
        self.y_train = self.mlb.fit_transform(
            self.reference_dataset['tags']
        ).astype(np.float32)

        self.tag_descriptor = self.build_tag_descriptor()

        print(self.tag_descriptor.head(n=20))

        self.num_classes = self.y_train[0].shape[0]

    def build_tag_descriptor(self):
        unlist_tags = []
        for _, row in self.reference_dataset.iterrows():
            unlist_tags += row['tags']

        tag_descriptor = pd.DataFrame(
            pd.Series(unlist_tags)
            .value_counts(sort=True, ascending=False)
        ).reset_index()
        tag_descriptor.columns = ['tag', 'freq']

        def get_index(tag):
            for i, t in enumerate(self.mlb.classes_):
                if t == tag:
                    return float(i)

        tag_descriptor['tag_index'] = (
            tag_descriptor['tag'].apply(get_index)
        )

        return tag_descriptor

    def read_reference_dataset(self, csv_path):
        tmp_df = (
            pd.read_csv(csv_path)
            .groupby('destination_path')['label']
            .apply(list)
        ).reset_index()
        tmp_df.columns = ['image', 'tags']
        print(tmp_df.head())
        return tmp_df
        # return tmp_df.head(n=300)

    def get_image(self, index):
        item_img_path = self.X_train[index]
        img = Image.open(item_img_path)
        img = img.convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img, item_img_path

    def get_labels(self, index):
        return self.y_train[index]

    def __getitem__(self, index):
        img, item_img_path = self.get_image(index)
        labels = self.get_labels(index)
        return img, labels, item_img_path

    def __len__(self):
        return len(self.X_train.index)


class TagwalkSequenceDataset(TagwalkDataset):
    def get_labels(self, index):
        tw_one_hot = self.y_train[index]
        tag_idx = np.where(tw_one_hot == 1)[0].tolist()

        # Ensure the sequence is ordered by tag frequency
        seq = (
            pd.DataFrame(
                self.mlb.classes_[tag_idx].tolist(),
                columns = ['tag']
            ).merge(self.tag_descriptor, on='tag', how='left')
            .sort_values('freq', ascending=False)
        )['tag_index'].values

        return torch.Tensor(seq)

    def __getitem__(self, index):
        img, _ = self.get_image(index)
        labels = self.get_labels(index)
        print(labels)
        return img, labels


def collate_sequence_data(data):
    """ Ensure all sequences have same length in the current batch
    """
    # Order batch by decreasing length
    data.sort(key=lambda x: len(x[1]), reverse=True)
    images, captions = zip(*data)
    print(captions)

    images = torch.stack(images, 0)
    lengths = [len(cap) for cap in captions]
    targets = torch.zeros(len(captions), max(lengths)).long()

    for i, cap in enumerate(captions):
        end = lengths[i]
        targets[i, :end] = cap[:end]
    return images, targets, lengths


def tagwalk_dataloader(dataset=None):
    if dataset is None:
        dataset = TagwalkDataset()
    train_loader = DataLoader(dataset,
                              batch_size=256,
                              shuffle=True,
                              num_workers=4)
    return train_loader


if __name__ == "__main__":
    TagwalkDataset()
