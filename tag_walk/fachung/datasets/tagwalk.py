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
        print BASE_PATH
        if csv_path is None:
            csv_path = BASE_PATH + '/assocs.csv'

        if img_path is None:
            img_path = BASE_PATH + 'images/all'

        self.reference_dataset = self.read_reference_dataset(csv_path)
        print self.reference_dataset.head()

        self.mlb = MultiLabelBinarizer()
        self.img_path = img_path
        self.transform = transform

        self.X_train = self.reference_dataset['image']
        self.y_train = self.mlb.fit_transform(
            self.reference_dataset['tags']
        ).astype(np.float32)

    def read_reference_dataset(self, csv_path):
        tmp_df = (
            pd.read_csv(csv_path)
            .groupby('image')['tag']
            .apply(list)
        ).reset_index()
        tmp_df.columns = ['image', 'tags']
        return tmp_df

    def __getitem__(self, index):
        item_img_path = '/'.join([
            self.img_path,
            self.X_train[index]
        ])
        img = Image.open(item_img_path)
        img = img.convert('RGB')
        if self.transform is not None:
            img = self.transform(img)

        label = torch.from_numpy(self.y_train[index])
        return img, label

    def __len__(self):
        return len(self.X_train.index)

def tagwalk_dataloader():
    tw_dataset = TagwalkDataset()
    train_loader = DataLoader(tw_dataset,
                              batch_size=256,
                              shuffle=True,
                              num_workers=4)
    return train_loader