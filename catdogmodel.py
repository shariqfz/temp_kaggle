import torch.nn as nn
import torch.nn.functional as F

class CatAndDogConvNet(nn.Module):

    def __init__(self):
        super().__init__()

        # onvolutional layers (3,16,32)
        self.conv1 = nn.Conv2d(in_channels = 3, out_channels = 16, kernel_size=(5, 5), stride=2, padding=1)
        self.conv2 = nn.Conv2d(in_channels = 16, out_channels = 32, kernel_size=(5, 5), stride=2, padding=1)
        self.conv3 = nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size=(3, 3), padding=1)

        # conected layers
        self.fc1 = nn.Linear(in_features= 64 * 6 * 6, out_features=500)
        self.fc2 = nn.Linear(in_features=500, out_features=50)
        self.fc3 = nn.Linear(in_features=50, out_features=1)


    def forward(self, X):

        X = F.relu(self.conv1(X))
        X = F.max_pool2d(X, 2)

        X = F.relu(self.conv2(X))
        X = F.max_pool2d(X, 2)

        X = F.relu(self.conv3(X))
        X = F.max_pool2d(X, 2)

        X = X.view(X.shape[0], -1)
        X = F.relu(self.fc1(X))
        X = F.relu(self.fc2(X))
        X = self.fc3(X)

        return X


import torch.nn as nn
import torch.nn.functional as F
import math

class CatAndDogConvNetCustom(nn.Module):

    def __init__(self, image_height, image_width):
        super().__init__()

        # Convolutional layers (3,16,32)
        self.conv1 = nn.Conv2d(in_channels = 3, out_channels = 16, kernel_size=(5, 5), stride=2, padding=1)
        self.conv2 = nn.Conv2d(in_channels = 16, out_channels = 32, kernel_size=(5, 5), stride=2, padding=1)
        self.conv3 = nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size=(3, 3), padding=1)

        # Calculate the output size after the convolutional and pooling layers
        self.size_after_conv1 = (199, 299) # (math.floor((image_height - 4) / 2) +1, math.floor((image_width - 4) / 2 )+1)
        self.size_after_pool1 = (99, 149)  # (math.ceil(self.size_after_conv1[0] / 2), math.ceil(self.size_after_conv1[1] / 2))
        self.size_after_conv2 = (49, 74)   # (math.ceil((self.size_after_pool1[0] - 4) / 2) +1, math.ceil((self.size_after_pool1[1] - 4) // 2 ) +1) # (49, 74)
        self.size_after_pool2 = (24, 37)   # (math.ceil(self.size_after_conv2[0] / 2), math.ceil(self.size_after_conv2[1] / 2))
        self.size_after_conv3 = (24, 37)   # (math.ceil((self.size_after_pool2[0] - 2) / 1) +1, math.ceil((self.size_after_pool2[1] - 2) / 1) +1)
        self.size_after_pool3 = (12, 18)   # (math.ceil(self.size_after_conv3[0] / 2), math.ceil(self.size_after_conv3[1] / 2))

        # Connected layers
        self.fc1 = nn.Linear(in_features= 64 * self.size_after_pool3[0] * self.size_after_pool3[1], out_features=500)
        self.fc2 = nn.Linear(in_features=500, out_features=50)
        self.fc3 = nn.Linear(in_features=50, out_features=1)


    def forward(self, X):

        # print(f"{X.shape = }")
        X = F.relu(self.conv1(X))
        # print(f"c1: {X.shape}\t{self.size_after_conv1}")
        X = F.max_pool2d(X, 2)
        # print(f"p1: {X.shape}\t{self.size_after_pool1}")

        X = F.relu(self.conv2(X))
        # print(f"c2: {X.shape}\t{self.size_after_conv2}")
        X = F.max_pool2d(X, 2)
        # print(f"p2: {X.shape}\t{self.size_after_pool2}")

        X = F.relu(self.conv3(X))
        # print(f"c3: {X.shape}\t{self.size_after_conv3}")
        X = F.max_pool2d(X, 2)
        # print(f"p3: {X.shape}\t{self.size_after_pool3}")

        X = X.view(X.shape[0], -1)
        # print(f"{X.shape = }")
        X = F.relu(self.fc1(X.reshape(X.shape[0],-1)))
        X = F.relu(self.fc2(X))
        X = self.fc3(X)

        return X