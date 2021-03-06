from random import randint

import scipy.spatial.distance as distance

import numpy as np
import pandas as pd

from sklearn.preprocessing import MultiLabelBinarizer

from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader

from PIL import Image

import fachung.configuration as conf
import fachung.logger as logger

from fachung.utils import from_numpy

BASE_PATH = (
    conf.BASE_DATA +
    conf.get_config()
    .get(conf.MODE, 'outputs')
)

RESTRICTIONS = [
    '%',
    'machinewash',
    'ourmodel'
]

# Clean that, get tag hierarchy for better similarity metrics
# Pure :hankey:
def str_to_array(s):
    try:
        arr = (
            s.replace('[', '')
            .replace(']', '')
            .replace('{', '')
            .replace('}', '')
            .replace("'", '')
            .replace('"', '')
            .replace(" ", '')
            .split(',')
        )
        filtered_array = []
        for element in arr:
            for restriction in RESTRICTIONS:
                if not restriction in element:
                    filtered_array.append(element)
        return filtered_array

    except AttributeError:
        return []

class AsosDataset(Dataset):
    def __init__(self, csv_path=None, img_path=None,
                 transform=None, mode='regression',
                 sim_threshold=0.8):
        if csv_path is None:
            csv_path = BASE_PATH + '/asos.csv'

        if img_path is None:
            img_path = BASE_PATH + '/asos_images'

        self.mode = mode
        self.sim_threshold = sim_threshold

        self.mlb = MultiLabelBinarizer()
        self.transform = transform

        self.reference_dataset = self.read_reference_dataset(csv_path)

        self.iids = self.reference_dataset['iid']
        self.X_train = self.reference_dataset['destination_path']

        logger.INFO("Fit/Transform MultiLabelBinarizer")
        self.labels = self.mlb.fit_transform(
            self.reference_dataset['labels']
        ).astype(np.float32)

        logger.INFO("Number of unique labels %s" %
                    (len(self.mlb.classes_.tolist())))

        self.img_path = img_path

    def get_image(self, index):
        item_img_path = self.X_train[index]
        img = Image.open(item_img_path)
        img = img.convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img, item_img_path

    def get_labels(self, index):
        return self.labels[index]

    def inner_iid_sim(self, index1, index2):
        iid1 = self.iids[index1]
        iid2 = self.iids[index2]
        if iid1 == iid2:
            return 1.0
        return -1.0

    def cosine_sim(self, index1, index2):
        iid1 = self.iids[index1]
        iid2 = self.iids[index2]
        if iid1 == iid2:
            return 1.0

        labels1 = self.get_labels(index1)
        labels2 = self.get_labels(index2)
        return 1.0 - distance.cosine(labels1, labels2)

    def thresholded_cosine(self, index1, index2):
        is_sim  = int(self.cosine_sim(index1, index2) >= self.sim_threshold)
        if not is_sim:
            return -1.0
        return 1.0

    def _get_similarity_fn(self):
        sim_fns = {
            'regression': self.cosine_sim,
            'thresholded_regression': self.thresholded_cosine
        }
        return sim_fns.get(self.mode, self.inner_iid_sim)

    def get_similarity(self, index1, index2):
        fn = self._get_similarity_fn()
        return fn(index1, index2)

    def read_reference_dataset(self, csv_path):
        logger.INFO("Reading reference_dataset")
        df = pd.read_csv(csv_path)
        df['labels'] = df['attributes'].apply(str_to_array)
        return df

    def __getitem__(self, index):
        img1, _ = self.get_image(index)

        index2 = randint(0, self.__len__())
        img2, _ = self.get_image(index2)

        similarity = self.get_similarity(index, index2)
        return img1, img2, similarity

    def __len__(self):
        return len(self.X_train.index)


def asos_siamese_dataloader(dataset=None):
    if dataset is None:
        dataset = AsosDataset()
    return DataLoader(dataset, batch_size=256,
                      shuffle=True, num_workers=4)

if __name__ == "__main__":
    dataset = AsosDataset()
    for i in range(50):
        print("---- %s ----" % i)
        print(dataset.__getitem__(11))
