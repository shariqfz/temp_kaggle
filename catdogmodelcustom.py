import torch.nn as nn
import torch.nn.functional as F

class CatAndDogConvNetCustom(nn.Module):

    def __init__(self):
        super().__init__()

        # Convolutional layers (3,16,32)
        self.conv1 = nn.Conv2d(in_channels = 3, out_channels = 16, kernel_size=(5, 5), stride=2, padding=1)
        self.conv2 = nn.Conv2d(in_channels = 16, out_channels = 32, kernel_size=(5, 5), stride=2, padding=1)
        self.conv3 = nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size=(3, 3), padding=1)

        # Connected layers
        # We'll initialize these in the forward pass
        self.fc1 = None
        self.fc2 = nn.Linear(in_features=500, out_features=50)
        self.fc3 = nn.Linear(in_features=50, out_features=1)


    def forward(self, X):

        X = F.relu(self.conv1(X))
        X = F.max_pool2d(X, 2)

        X = F.relu(self.conv2(X))
        X = F.max_pool2d(X, 2)

        X = F.relu(self.conv3(X))
        X = F.max_pool2d(X, 2)

        # If the fully connected layer is not yet initialized
        if self.fc1 is None:
            # Calculate the number of flattened features
            num_flatten = X.shape[1] * X.shape[2] * X.shape[3]
            # Initialize the fully connected layer
            self.fc1 = nn.Linear(num_flatten, 500)

        # Flatten the tensor
        X = X.view(X.shape[0], -1)

        X = F.relu(self.fc1(X))
        X = F.relu(self.fc2(X))
        X = self.fc3(X)

        return X
