import numpy as np
from sklearn.metrics import f1_score

from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim
from torch.autograd import Variable

import torchvision
import torchvision.transforms as transforms

import fachung.logger as logger
from fachung.datasets.tagwalk import TagwalkDataset
from fachung.utils import get_train_valid_test_loaders
from fachung.models.trainer import Trainer

NORMALIZE = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

DEFAULT_TRANSFORMS = (
    transforms.Compose([
        # transforms.RandomSizedCrop(224),
        # transforms.RandomHorizontalFlip(),
        transforms.Scale((224, 224)),
        transforms.ToTensor(),
        NORMALIZE
    ])
)


def f1_batch(pred, ground):
    f1 = np.empty((pred.shape[0], 1), dtype='float32')
    for i in range(pred.shape[0]):
        f1[i] = f1_score(ground[i], pred[i])
    return f1


BASIC_CONFIG = {
    'batch_size': 4,
    'num_epochs': 2,
    'output_dir': './data/training_logs',
    'debug': True,
    'reset': False,
    'data_path': 'data/tag_walk/',
    'model_id': "ttttt"
}

class TagWalkClassifier(Trainer):

    @property
    def classes(self):
        return self.dataset.mlb.classes_

    @property
    def num_classes(self):
        return len(self.classes)

    @property
    def criterion(self):
        return nn.BCEWithLogitsLoss()
        # return nn.MultiLabelSoftMarginLoss()

    @property
    def optimiser(self):
        return optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9)

    def build_dataset(self):
        return TagwalkDataset(
            csv_path=self.data_path + 'assocs.csv',
            img_path=self.data_path + 'images/all',
            transform=DEFAULT_TRANSFORMS
        )

    def build_model(self):
        model = torchvision.models.resnet50(pretrained=True)
        num_classes = self.dataset.num_classes
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model

    def split_dataset(self):
        train_loader, validation_loader, testing_loader = (
            get_train_valid_test_loaders(self.dataset, self.batch_size, 42)
        )

        return {
            'training': train_loader,
            'validation': validation_loader,
            'testing': testing_loader
        }

    def show_debug(self, batch_data):
        image = Variable(batch_data[0])
        target = Variable(batch_data[1])
        img_path = batch_data[2]

        output = self.model(image)

        output_mat = output.data.cpu().numpy()
        target_mat = target.data.cpu().numpy()
        for batch_id in range(output_mat.shape[0]):
            print(batch_id)
            max_idx = np.argpartition(output_mat[batch_id], 10)[:10]
            print("IMG: %s \n Predicted: %s \n Truth: %s" % (
                img_path[batch_id],
                self.classes[max_idx],
                self.classes[
                    np.where(target_mat[batch_id] == 1)
                ]))

    def on_batch_data(self, batch_data, mode):
        image = Variable(batch_data[0])
        target = Variable(batch_data[1])

        output = self.model(image)
        loss = self.criterion(output, target)

        return output, loss, {}

    def update_metrics(self, output, target, mode='training'):
        output_mat = F.sigmoid(output).data.cpu().numpy()
        target_mat = target.data.cpu().numpy()
        f1_score = f1_batch(target_mat, output_mat.round())
        key = 'f1'
        if mode != 'training':
            key = '_'.join([mode, key])
        self.history['metrics'][key].append(f1_score)
        return f1_score


if __name__ == "__main__":
    classifier = TagWalkClassifier(BASIC_CONFIG)
    history = classifier.run(training=True)
    # print(history)
