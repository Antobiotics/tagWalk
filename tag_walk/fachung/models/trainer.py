import pandas as pd
import numpy as np
from tqdm import tqdm

import torch

import fachung.logger as logger


def save_checkpoint(state, filename='checkpoint.pth.tar'):
    torch.save(state, filename)


class Trainer():
    def __init__(self, config):
        self.config = config

        self.debug = self.config.get('debug', True)
        self.reset = self.config.get('reset', False)

        self.batch_size = self.config.get('batch_size', 4)
        self.num_epochs = self.config.get('num_epochs', 2)

        self.data_path = self.config['data_path']
        self.output_dir = self.config.get('output_dir', './data/training_logs')

        model_id = self.config.get('model_id', '')
        if self.debug:
            model_id = 'debug'
        self.model_id = model_id

        self.dataset = self.build_dataset()
        self._model = self.build_model()

        self.loader_dict = self.split_dataset()

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
        raise RuntimeError("self.criterion must be set")

    @property
    def optimiser(self):
        raise RuntimeError("self.optimiser must be set")

    @property
    def metrics_names(self):
        return ['loss']

    def read_model(self):
        logger.INFO("Trying to load %s" % (self.chk_filename))
        checkpoint = torch.load(self.chk_filename)
        self.model.load_state_dict(checkpoint['state_dict'])
        self.history = checkpoint['history']

    def build_model(self):
        raise RuntimeError("build_model must be implemented")

    def build_dataset(self):
        raise RuntimeError("build_dataset must be implemented")

    def split_dataset(self):
        raise RuntimeError("split_dataset must be implemented")

    def init(self):
        if self.reset:
            self.init_history()
        else:
            try:
                self.read_model()
            except Exception as e:
                logger.WARN("No previous model found")
                logger.ERROR(e)
                self.init_history()

    def init_history_metrics(self, metrics_name):
        self.history['metrics'][metrics_name] = []
        self.history['metrics']['val_' + metrics_name] = []

    def init_history(self):
        self.history = {
            'current_epoch': 0,
            'batch_size': self.batch_size,
            'num_epoch': self.num_epochs,
            'metrics': {}
        }
        for m_name in self.metrics_names:
            self.init_history_metrics(m_name)

    def update_history_metrics(self, metrics):
        for m_name in metrics:
            self.history['metrics'][m_name].appen(metrics[m_name])

    def must_save(self, epoch, loss):
        losses_df = pd.DataFrame({'loss': self.history['metrics']['val_loss']})
        losses_df['index'] = losses_df.index
        means = losses_df['loss'].rolling(self.batch_size).mean().tolist()
        return (
            epoch == 0 or
            means[-1] <= means[-2]
        )

    def show_progress(self, pbar, epoch, losses, metrics):
        mean_loss = np.mean(np.array(losses))
        desc_fmt = 'Epoch: {}; Loss: {:.5f}; Avg: {:.5f}'
        (
            pbar
            .set_description(desc_fmt.format(epoch + 1,
                                             losses[-1],
                                             mean_loss,))
        )

    def show_debug(self, batch_data):
        pass

    def on_batch_data(self, batch_data, mode):
        pass

    def train(self, epoch):
        dataset = iter(self.loader_dict['training'])
        pbar = tqdm(dataset)

        self.model.train(True)

        losses = []
        for _, batch_data in enumerate(pbar):
            self.optimiser.zero_grad()

            _ , loss, metrics = self.on_batch_data(batch_data,
                                                   mode='training')
            self.update_history_metrics(metrics)

            loss.backward()
            self.optimiser.step()

            losses.append(loss.data[0])
            self.show_progress(pbar, epoch, losses, metrics)

            if self.debug:
                self.show_debug(batch_data)


    def validate(self, epoch):
        dataset = iter(self.loader_dict['validation'])
        pbar = tqdm(dataset)

        self.model.eval()

        losses = []
        for _, batch_data in enumerate(pbar):
            _ , loss, metrics = self.on_batch_data(batch_data,
                                                   mode='validation')
            self.update_history_metrics(metrics)
            losses.append(loss.data[0])
            self.show_debug(batch_data)

        mean_loss = np.mean(np.array(losses))
        if self.must_save(epoch, mean_loss):
            logger.INFO("Saving for validation loss %s" % (mean_loss))
            save_checkpoint({
                'history': self.history,
                'state_dict': self.model.state_dict()
            }, filename=self.chk_filename)

    def fit(self):
        self.init()
        start_epoch = self.history['current_epoch']
        pbar = tqdm(range(start_epoch, self.num_epochs + 1))

        for epoch in pbar:
            self.train(epoch)
            self.validate(epoch)
            self.history['current_epoch'] = epoch + 1

        return self.history

    def run(self, training=True):
        if training:
            return self.fit()
        return None
