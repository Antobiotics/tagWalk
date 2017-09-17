import torch
import torch.optim as optim

import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence

from torch.autograd import Variable

import torchvision.models as models

import fachung.datasets.tagwalk as tw_data
import fachung.utils as utils
import fachung.transforms as transforms

from fachung.models.trainer import Trainer


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        """Load the pretrained ResNet-152 and replace top fc layer."""
        super(EncoderCNN, self).__init__()
        resnet = models.resnet152(pretrained=True)
        modules = list(resnet.children())[:-1]      # delete the last fc layer.
        self.resnet = nn.Sequential(*modules)
        self.linear = nn.Linear(resnet.fc.in_features, embed_size)
        self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)
        self.init_weights()

    def init_weights(self):
        """Initialize the weights."""
        self.linear.weight.data.normal_(0.0, 0.02)
        self.linear.bias.data.fill_(0)

    def forward(self, images):
        """Extract the image feature vectors."""
        features = self.resnet(images)
        features = Variable(features.data)
        features = features.view(features.size(0), -1)
        features = self.bn(self.linear(features))
        return features


class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers):
        """Set the hyper-parameters and build the layers."""
        super(DecoderRNN, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.init_weights()

    def init_weights(self):
        """Initialize weights."""
        self.embed.weight.data.uniform_(-0.1, 0.1)
        self.linear.weight.data.uniform_(-0.1, 0.1)
        self.linear.bias.data.fill_(0)

    def forward(self, features, captions, lengths):
        """Decode image feature vectors and generates captions."""
        embeddings = self.embed(captions)
        embeddings = torch.cat((features.unsqueeze(1), embeddings), 1)
        packed = pack_padded_sequence(embeddings, lengths, batch_first=True)
        hiddens, _ = self.lstm(packed)
        outputs = self.linear(hiddens[0])
        return outputs

    def sample(self, features, states=None):
        """Samples captions for given image features (Greedy search)."""
        sampled_ids = []
        inputs = features.unsqueeze(1)
        for _ in range(20):                                      # maximum sampling length
            hiddens, states = self.lstm(inputs, states)          # (batch_size, 1, hidden_size),
            outputs = self.linear(hiddens.squeeze(1))            # (batch_size, vocab_size)
            predicted = outputs.max(1)[1]
            sampled_ids.append(predicted)
            inputs = self.embed(predicted)
        sampled_ids = torch.cat(sampled_ids, 1)                  # (batch_size, 20)
        return sampled_ids.squeeze()

class TagWalkCNNRNN(Trainer):
    @property
    def embedding_size(self):
        return self.options.get('embedding_size', 32)

    @property
    def hidden_size(self):
        return self.options.get('hidden_size', 256)

    @property
    def num_layers(self):
        return self.options.get('num_layers', 4)

    @property
    def classes(self):
        return self.dataset.mlb.classes_

    @property
    def num_classes(self):
        return len(self.classes)

    @property
    def criterion(self):
        return nn.CrossEntropyLoss()

    @property
    def optimiser(self):
        parameters = (
            list(self.model['decoder'].parameters()) +
            list(self.model['encoder'].linear.parameters()) +
            list(self.model['encoder'].bn.parameters())
        )
        return optim.Adam(parameters, lr=self.learning_rate)

    def build_dataset(self):
        return tw_data.TagwalkSequenceDataset(
            csv_path=self.data_path + 'assocs.csv',
            img_path=self.data_path + 'images/all',
            transform=transforms.DEFAULT_TRANSFORMS
        )

    def build_model(self):
        encoder = EncoderCNN(self.embedding_size)
        decoder = DecoderRNN(self.embedding_size, self.hidden_size,
                             self.num_classes, self.num_layers)
        return {
            'encoder': encoder,
            'decoder': decoder
        }

    def split_dataset(self):
        train_loader, validation_loader, testing_loader = (
            utils
            .get_train_valid_test_loaders(self.dataset,
                                          self.batch_size,42,
                                          collate_fn=tw_data.collate_sequence_data)
        )

        return {
            'training': train_loader,
            'validation': validation_loader,
            'testing': testing_loader
        }

    def show_debug(self, batch_data):
        pass

    def on_batch_data(self, batch_data, mode):
        images = Variable(batch_data[0])
        labels = Variable(batch_data[1])
        lengths = batch_data[2]

        targets = pack_padded_sequence(labels, lengths, batch_first=True)[0]

        features = self.model['encoder'](images)
        outputs = self.model['decoder'](features, labels, lengths)
        loss = self.criterion(outputs, targets)

        self.update_loss_history(loss, mode)

        return outputs, loss, {}
