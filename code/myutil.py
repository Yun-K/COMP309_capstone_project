
# baseline MLP model


from torch import nn
from torch.nn.modules.container import Sequential

import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self):
        super().__init__()

        self.layers = Sequential(
            # nn.Conv2d(32, (3, 3), input_shape=(300, 300, 3),
            #           activation='sigmoid', padding='same'),
            nn.Flatten(),
            nn.Linear(270000, 64),
            # https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md
            # nn.MaxPool2d(),
            # nn.Softshrink(),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),

            # 3 classes, so final out_chanel : 3
            nn.Linear(32, 3)

        )

    def forward(self, X):
        """Do the forward pass on MLP

        Args:
            X ([type]): [description]

        """
        return self.layers(X)

    def to_model_string(self):
        return 'MLP'


# -----------------------------------------------------------------
# -----------------------------------------------------------------
# https://www.kaggle.com/artgor/simple-eda-and-model-in-pytorch/notebook


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 2)
        self.dropout = nn.Dropout(0.2)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 64 * 4 * 4)
        #x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.sig(self.fc3(x))
        return x

    def to_model_string(self):
        return 'CNN'


# # # get some random training images
# # dataiter = iter(trainloader)
# # images, labels = dataiter.next()

# # # show images
# # imshow(torchvision.utils.make_grid(images))
# # # print labels
# # print(' '.join('%5s' % classes[labels[j]] for j in range(batch_size)))
