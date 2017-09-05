import json
import numpy as np

from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

import torchvision
import torchvision.transforms as transforms

from torchsample.modules import ModuleTrainer
import torchsample.callbacks as callbacks

import fachung.logger as logger
from fachung.datasets.tagwalk import TagwalkDataset
from fachung.utils import get_train_valid_test_loaders


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


def save_checkpoint(state, filename='checkpoint.pth.tar'):
    torch.save(state, filename)


class TagWalkClassifier():

    def __init__(self, data_path, batch_size=4, num_epochs=2,
                 debug=False, reset=False):
        self.debug = debug
        self.reset = reset
        self.data_path = data_path
        self.batch_size = batch_size
        self.num_epochs = num_epochs

        self.dataset = self.build_dataset()

        self.train_loader, self.validation_loader, self.testing_loader = (
            self.split_dataset()
        )

        self._model = self.build_model()

    @property
    def output_dir(self):
        return './data/training_logs'

    @property
    def model_name(self):
        return self.__class__.__name__

    @property
    def chk_filename(self):
        return '/'.join([
            self.output_dir,
            self.model_name + '.pkl'
        ])

    @property
    def callbacks(self):
        return [
            callbacks.EarlyStopping(monitor='val_loss', patience=5),
            callbacks.ModelCheckpoint(directory='./data/trainer_logs'),
            callbacks.CSVLogger('./data/trainer_logs/tw_history.csv')
        ]

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, value):
        self._model = value

    @property
    def criterion(self):
        return nn.BCEWithLogitsLoss()
        # return nn.MultiLabelSoftMarginLoss()

    @property
    def optimiser(self):
        return optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9)

    def init_history(self):
        self.history = {
            'current_epoch': 0,
            'batch_size': self.batch_size,
            'num_epoch': self.num_epochs,
            'loss': [],
            'val_loss': [],
            'loss_epoch': [],
            'val_loss_epoch': []
        }

    def read_model(self):
        logger.INFO("Trying to load %s" % (self.chk_filename))
        checkpoint = torch.load(self.chk_filename)
        self.model.load_state_dict(checkpoint['state_dict'])
        self.history = checkpoint['history']

    def build_dataset(self):
        return TagwalkDataset(
            csv_path=self.data_path + 'assocs.csv',
            img_path=self.data_path + 'images/all',
            transform=DEFAULT_TRANSFORMS
        )

    def build_model(self):
        model = torchvision.models.resnet18(pretrained=True)
        num_classes = self.dataset.num_classes
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model

    def split_dataset(self):
        return get_train_valid_test_loaders(self.dataset, self.batch_size, 42)

    def build_trainer(self):
        trainer = ModuleTrainer(self.model)
        trainer.compile(loss='multilabel_soft_margin_loss',
                        optimizer='adadelta')
        return trainer

    def show_debug(self, output, target, img_path):
        output_mat = output.data.cpu().numpy()
        target_mat = target.data.cpu().numpy()

        for batch_id in range(self.batch_size):
            max_idx = np.argpartition(output_mat[batch_id], 5)[:5]
            print("IMG: %s \n Predicted: %s \n Truth: %s" % (
                img_path[batch_id],
                self.dataset.mlb.classes_[max_idx],
                self.dataset.mlb.classes_[
                    np.where(target_mat[batch_id] == 1)
                ])
            )

    def train(self, epoch):
        dataset = iter(self.train_loader)
        pbar = tqdm(dataset)

        self.model.train(True)

        losses = []
        for input_image, target, img_path in pbar:
            image = Variable(input_image)
            target = Variable(target)

            output = self.model(image)
            loss = self.criterion(output, target)

            if self.debug:
                self.show_debug(output, target, img_path)

            loss.backward()
            self.optimiser.step()

            loss_value = loss.data[0]
            losses.append(loss_value)
            mean_loss = np.mean(np.array(losses))

            (
                pbar
                .set_description('Epoch: {}; Loss: {:.5f}; Avg: {:.5f}'.format(
                    epoch + 1, loss_value, mean_loss))
            )
            self.history['loss'].append(loss_value)
        self.history['loss_epoch'].append(np.mean(np.array(losses)))

    def validate(self, epoch):
        dataset = iter(self.validation_loader)
        pbar = tqdm(dataset)

        self.model.train(False)

        losses = []
        for i, val_data in enumerate(pbar):
            image = Variable(val_data[0])
            target = Variable(val_data[1])

            output = self.model(image)
            val_loss = self.criterion(output, target)
            val_loss_value = val_loss.data[0]

            losses.append(val_loss_value)
            self.history['val_loss'].append(val_loss_value)

        mean_val_loss = np.mean(np.array(losses))
        if self.must_save(mean_val_loss):
            # torch.save(self.model.state_dict(), self.chk_filename)
            save_checkpoint({
                'history': self.history,
                'state_dict': self.model.state_dict()
            }, filename=self.chk_filename)

        self.history['val_loss_epoch'].append(mean_val_loss)

    def must_save(self, loss):
        return (
            self.history['val_loss_epoch'] == [] or
            loss <= min(self.history['val_loss_epoch'])
        )

    def fit(self):
        if self.reset:
            self.init_history()
        else:
            try:
                self.read_model()
            except Exception as e:
                logger.INFO("No previous model found")
                print(e)
                self.init_history()

        start_epoch = self.history['current_epoch']
        pbar = tqdm(range(start_epoch, self.num_epochs))

        for epoch in pbar:
            self.train(epoch)
            self.validate(epoch)

            self.history['current_epoch'] = epoch + 1

        return self.history

    def run(self, training=True, reset=True):
        if training:
            return self.fit()
        return None


if __name__ == "__main__":
    classifier = TagWalkClassifier('data/tag_walk/',
                                   batch_size=4, num_epochs=2,
                                   debug=False, reset=False)
    history = classifier.run(training=True)
    print(history)
