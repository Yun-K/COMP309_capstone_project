# TODO: need to install torchmetric for accuracy
# !pip install torchmetrics

import os
import time

# import cv2
# import helper
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch  # PyTorch
import torch.autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms  # for image Transformation
import tqdm
from PIL import Image
from sklearn.datasets import load_digits
from sklearn.metrics import (accuracy_score, confusion_matrix, make_scorer,
                             mean_absolute_error, mean_squared_error,
                             precision_recall_fscore_support, recall_score,
                             roc_auc_score)
from sklearn.model_selection import (KFold, LeaveOneOut, ShuffleSplit,
                                     cross_val_score, cross_validate,
                                     train_test_split)
from sklearn.neural_network import MLPClassifier, MLPRegressor
from torch import nn
from torch.autograd import Variable
from torch.nn.modules.container import Sequential
from torch.optim import lr_scheduler
from torch.optim.lr_scheduler import (CosineAnnealingLR, ReduceLROnPlateau,
                                      StepLR)
from torch.utils.data import DataLoader, Dataset, TensorDataset, random_split
from torch.utils.data.sampler import SubsetRandomSampler  # Sampler
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
from torchvision.datasets import ImageFolder

# CNN model
# # https://www.kaggle.com/artgor/simple-eda-and-model-in-pytorch/notebook
# # https://www.mashen.zone/thread-1825047.htm
# # https://blog.csdn.net/nanke_4869/article/details/113458729
# # CNN model
# something has to be commented out since no enough GPU memory in Colab


class CNN(nn.Module):
    """
  https://datascience.stackexchange.com/questions/40906/determining-size-of-fc-layer-after-conv-layer-in-pytorch
    https://towardsdatascience.com/classification-of-fruit-images-using-neural-networks-pytorch-1d34d49342c7
    """

    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3,
                      stride=1, padding=1),  # 3 channels to 32 channels
            # Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),  # activitation function
            # output: 32 channels x 150 x 150 image size - decrease
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.25),

            nn.Conv2d(in_channels=32, out_channels=64,
                      kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),  # activitation function
            nn.MaxPool2d(2, 2),  # output: 64 x 75 x 75
            # use dropout to enable a kind of early stop which can avoid the over-fit
            nn.Dropout(0.25),

            nn.Conv2d(in_channels=64, out_channels=128,
                      kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),  # activitation function
            # can keep the same, increase power of model , go deeper as u add linearity to non-linearity
            nn.Conv2d(in_channels=128, out_channels=128,
                      kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),  # activitation function
            nn.MaxPool2d(3, 3),  # output: 128 x 25 x 25
            # use dropout to enable a kind of early stop which can avoid the over-fit
            nn.Dropout(0.25),

            nn.Conv2d(in_channels=128, out_channels=256,
                      kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),  # activitation function
            nn.Conv2d(in_channels=256, out_channels=256,
                      kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),  # activitation function
            nn.Dropout(0.25),

            nn.Conv2d(in_channels=256, out_channels=512,
                      kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            # nn.MaxPool2d(2),
            nn.ReLU(inplace=True),  # activitation function
            # nn.Conv2d(in_channels=256, out_channels= 256, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm2d(512),
            # nn.MaxPool2d(2),
            # nn.ReLU(inplace = True), # activitation function
            nn.MaxPool2d(5, 5),  # output: 512 x 5 x 5
            # use dropout to enable a kind of early stop which can avoid the over-fit
            nn.Dropout(0.25),

            # nn.Conv2d(in_channels=256, out_channels= 512, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm2d(512),
            # nn.ReLU(inplace = True), # activitation function
            # # nn.Conv2d(in_channels=512, out_channels= 512, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm2d(512),
            # # nn.MaxPool2d(2),
            # nn.ReLU(inplace = True), # activitation function
            # nn.MaxPool2d(5, 5), # output: 512 x 5 x 5
            # # use dropout to enable a kind of early stop which can avoid the over-fit
            # nn.Dropout(0.25),

            nn.Flatten(),  # a single vector 512*5*5,
            nn.Linear(512*5*5, 512),
            nn.ReLU(inplace=True),  # activitation function
            nn.Dropout(0.25),

            # nn.ReLU(inplace = True), # activitation function
            # nn.Linear(1024, 512),
            # nn.ReLU(),
            nn.Linear(512, 3)


            # nn.Dropout(0.25),
            # 512,131 on towardsDatascience, dont know what it is ,
            # I think it might be the batch_size(128) + classes(3)?
            # need to test performances
            # nn.Linear(512, 131)

        )

    def forward(self, xb):
        return self.network(xb)

    def to_model_string(self):
        return 'CNN'


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DEVICE = torch.device(DEVICE)
print("we can use:", DEVICE, "to run the Model ")

DATA_PATH = "testdata/"
MODEL_SAVE_PATH = "model.pth"

model = CNN()
model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()


def load_preprocess_construct_data(DATA_PATH=DATA_PATH, batch_size=64):
    dataset_whole = ImageFolder(DATA_PATH, transform=transforms.Compose(
        [
            # scale each images into the same size
            transforms.Resize((300, 300)),
            transforms.ToTensor(),  # transform them into tensor

            # normalize tensor images with mean and std
            # which means all channel of the input tensor images will be normalized
            transforms.Normalize(mean=(0.5, 0.5, 0.5),
                                 std=(0.5, 0.5, 0.5), inplace=False),
            # transforms.RandomRotation(degrees=(0, 180))
        ])
    )

    # shuffle:TRUE means random sample
    # drop_last:False will make sure that no images will be droped even there are no enough BATCH_SIZE(e.g. 64) images at the last round
    whole_loader = DataLoader(
        dataset_whole, batch_size=batch_size,  # num_workers=2,
        shuffle=True, drop_last=False)

    # print(dataset_whole.classes)
    print("whole set len:", len(dataset_whole))

    return dataset_whole,  whole_loader


dataset_whole,  whole_loader = load_preprocess_construct_data()

val_loss, loss_function = [], nn.CrossEntropyLoss().to(DEVICE)


def get_accuracy(model=model, testloader=whole_loader):
    """ 
    this code snipest is from:
    https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
    """
    correct, total = 0, 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)
            # calculate outputs by running images through the network
            outputs = model(images)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            loss = loss_function(outputs, labels)
            val_loss.append(loss.item())
    final_valid_loss_mean = np.sum(val_loss)/len(dataset_whole)

    final_acc = 100 * correct / total
    return final_acc, correct, total, final_valid_loss_mean


final_acc, correct_number, total_number, final_valid_loss_mean = get_accuracy()

print('-'*60)
print(f"There are {correct_number} images are classified correctly out of the total {total_number} images. \
        \n Final valid loss:{final_valid_loss_mean:7f} \
        \n Final acc:{final_acc:.5f}%")
