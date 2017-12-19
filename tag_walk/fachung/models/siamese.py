
import torch.nn as nn
import torchvision.models as models

from fachung.utils import Variable


class SiameseNetwork(nn.Module):
    def __init__(self, embed_size):
        super(SiameseNetwork, self).__init__()
        resnet = models.resnet152(pretrained=True)
        modules = list(resnet.children())[:-1]

        self.cnn = nn.Sequential(*modules)

        self.fc = nn.Sequential(
            nn.Linear(resnet.fc.in_features, embed_size),
            nn.ReLU(inplace=True),

            nn.Linear(embed_size, embed_size),
            nn.ReLU(inplace=True),

            nn.Linear(embed_size, 1)
        )

    def forward_once(self, x):
        features = self.cnn(x)
        features = Variable(features.data)
        features = features.view(features.size(0), -1)
        output = self.fc(features)
        return output

    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        return output1, output2
