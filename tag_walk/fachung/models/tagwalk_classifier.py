import torch.nn as nn
import torch.optim as optim

import torchvision
import torchvision.transforms as transforms

from inferno.trainers.basic import Trainer
from inferno.trainers.callbacks.logging.tensorboard import TensorboardLogger

from fachung.datasets.tagwalk import TagwalkDataset
from fachung.utils import get_train_valid_test_loader


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


class TagWalkClassifier():

    def __init__(self, data_path, batch_size=4, num_epochs=2):
        self.data_path = data_path
        self.batch_size = batch_size
        self.num_epochs = num_epochs

        self.dataset = self.build_dataset()

        self.train_loader, self.validation_loader, self.testing_loader = (
            self.split_dataset()
        )

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
        return get_train_valid_test_loaders(tw_dataset, self.batch_size, 42)

    def build_trainer(self):
        trainer = (
            Trainer(self.build_model())
            .build_criterion('MultiLabelSoftMarginLoss')
            .build_metric('CategoricalError')
            .build_optimizer('Adam')
            .validate_every((1, 'epochs'))
            .save_every((1, 'epochs'))
            .save_to_directory('../data/trainer_logs')
            .set_max_num_epochs(self.num_epochs)
            .build_logger(TensorboardLogger(log_scalars_every=(1, 'iteration'),
                                            log_images_every=(1, 'epochs')),
                          log_directory='../data/tf_logging')
        )

        return (
            trainer
            .bind_loader('train', self.train_loader)
            .bind_loader('validate', self.validation_loader)
        )

    def fit(self):
        trainer = self.build_trainer()
        trainer.fit()

        return trainer()

    def run(self, training=True):
        if training:
            return self.fit()
        return None
