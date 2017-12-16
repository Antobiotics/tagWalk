import torch.nn as nn
import torch.optim as optim

import fachung.datasets.asos as asos_data
import fachung.utils as utils
from fachung.utils import Variable

from fachung.models.siamese import SiameseNetwork
from fachung.models.trainer import Trainer

class AsosSiameseTrainer(Trainer):

    @property
    def embedding_size(self):
        return self.options.get('embedding_size', 128)

    @property
    def criterion(self):
        return nn.MSELoss()

    @property
    def optimiser(self):
        return optim.SGD(self.model['model'].paramters(),
                         lr=self.learning_rate,
                         momentum=0.9)

    def build_dataset(self):
        return asos_data.AsosDataset()

    def build_model(self):
        return {'model': SiameseNetwork(self.embedding_size)}

    def split_dataset(self):
        train_loader, validation_loader, testing_loader = (
            utils
            .get_train_valid_test_loaders(self.dataset,
                                          self.batch_size,42)
        )

        return {
            'training': train_loader,
            'validation': validation_loader,
            'testing': testing_loader
        }

    def show_debug(self, batch_data):
        pass

    def on_batch_data(self, batch_data, mode):
        image1 = Variable(batch_data[0])
        image2 = Variable(batch_data[1])
        similarity = batch_data[2]

        output1, output2 = self.model['model'](image1, image2)
        loss = self.criterion(output1, output2, similarity)

        self.update_loss_history(loss, mode)
        return _, loss, {}
