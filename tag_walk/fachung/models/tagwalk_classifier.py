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


def f1_batch(pred, ground):
    f1 = np.empty((pred.shape[0], 1), dtype='float32')
    for i in range(pred.shape[0]):
        f1[i] = f1_score(ground[i], pred[i])
    return f1


class TagWalkClassifier():

    def __init__(self, data_path, batch_size=4, num_epochs=2,
                 debug=False, reset=False, output_dir='./data/training_logs',
                 model_id=''):
        self.debug = debug
        self.reset = reset
        self.output_dir = output_dir

        self.data_path = data_path
        self.batch_size = batch_size
        self.num_epochs = num_epochs

        self.dataset = self.build_dataset()

        self.train_loader, self.validation_loader, self.testing_loader = (
            self.split_dataset()
        )

        self._model = self.build_model()
        self.classes = self.dataset.mlb.classes_
        self.num_classes = len(self.classes)

        if debug:
            model_id = 'debug'
        self.model_id = model_id

    @property
    def model_name(self):
        return self.__class__.__name__

    @property
    def chk_filename(self):
        return '/'.join([
            self.output_dir,
            self.model_name + '__' + self.model_id + '.pkl'
        ])

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
            'f1': [],
            'loss_epoch': [],
            'val_loss_epoch': [],
            'f1_epoch': [],
            'val_f1': [],
            'val_f1_epoch': []
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
        model = torchvision.models.resnet50(pretrained=True)
        num_classes = self.dataset.num_classes
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model

    def split_dataset(self):
        return get_train_valid_test_loaders(self.dataset, self.batch_size, 42)

    def show_debug(self, output, target, img_path):
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
                ])
            )

    def update_metrics(self, output, target, mode='training'):
        output_mat = F.sigmoid(output).data.cpu().numpy()
        target_mat = target.data.cpu().numpy()
        f1_score = f1_batch(target_mat, output_mat.round())
        key = 'f1'
        if mode != 'training':
            key = '_'.join([mode, key])
        self.history[key].append(f1_score)
        return f1_score

    def train(self, epoch):
        dataset = iter(self.train_loader)
        pbar = tqdm(dataset)

        self.model.train(True)

        losses = []
        f1_scores = []
        for input_image, target, img_path in pbar:
            image = Variable(input_image)
            target = Variable(target)

            self.optimiser.zero_grad()
            output = self.model(image)
            loss = self.criterion(output, target)

            if self.debug:
                self.show_debug(output, target, img_path)

            f1_score = self.update_metrics(output, target, mode='training')
            f1_scores.append(f1_score)
            mean_f1 = np.mean(np.array(f1_scores))

            loss.backward()
            self.optimiser.step()

            loss_value = loss.data[0]
            losses.append(loss_value)
            mean_loss = np.mean(np.array(losses))

            desc_fmt = 'Epoch: {}; Loss: {:.5f}; Avg: {:.5f}; F1: {:.5f}'
            (
                pbar
                .set_description(desc_fmt.format(epoch + 1,
                                                 loss_value,
                                                 mean_loss,
                                                 mean_f1))
            )

            self.history['loss'].append(loss_value)
        self.history['loss_epoch'].append(mean_loss)
        self.history['f1_epoch'].append(mean_f1)

    def validate(self, epoch):
        dataset = iter(self.validation_loader)
        pbar = tqdm(dataset)

        self.model.eval()

        losses = []
        f1_scores = []
        for i, val_data in enumerate(pbar):
            image = Variable(val_data[0])
            target = Variable(val_data[1])

            output = self.model(image)
            val_loss = self.criterion(output, target)
            val_loss_value = val_loss.data[0]
            losses.append(val_loss_value)

            self.show_debug(F.sigmoid(output), target, val_data[2])
            self.history['val_loss'].append(val_loss_value)

            f1_score = self.update_metrics(output, target, mode='val')
            f1_scores.append(f1_score)

        mean_val_loss = np.mean(np.array(losses))
        mean_f1 = np.mean(np.array(f1_scores))

        if self.must_save(mean_val_loss):
            logger.INFO("Saving for validation loss %s" % (mean_val_loss))
            save_checkpoint({
                'history': self.history,
                'state_dict': self.model.state_dict()
            }, filename=self.chk_filename)

        self.history['val_loss_epoch'].append(mean_val_loss)
        self.history['val_f1_epoch'].append(mean_f1)

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
        pbar = tqdm(range(start_epoch, self.num_epochs + 1))

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
                                   batch_size=4, num_epochs=3,
                                   debug=False, reset=False, model_id='d3bb959')
    history = classifier.run(training=True)
    # print(history)
