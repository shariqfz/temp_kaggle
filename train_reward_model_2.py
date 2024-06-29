import torch
device = 'cuda' if torch.cuda.is_available() else 'cpu'

import random
random.seed(42)
torch.manual_seed(42)

import pandas as pd

import torch
from torch import nn as nn
import numpy as np


def fanin_init(tensor):
    size = tensor.size()
    if len(size) == 2:
        fan_in = size[0]
    elif len(size) > 2:
        fan_in = np.prod(size[1:])
    else:
        raise Exception("Shape must be have dimension at least 2.")
    bound = 1. / np.sqrt(fan_in)
    return tensor.data.uniform_(-bound, bound)

class CNN(nn.Module):
    def __init__(
            self,
            input_width,
            input_height,
            input_channels,
            output_size,
            kernel_sizes,
            n_channels,
            strides,
            paddings,
            hidden_sizes=None,
            added_fc_input_size=0,
            batch_norm_conv=False,
            batch_norm_fc=False,
            init_w=1e-4,
            hidden_init=nn.init.xavier_uniform_,
            hidden_activation=nn.ReLU(),
            output_activation=nn.Tanh(),
    ):
        if hidden_sizes is None:
            hidden_sizes = []
        assert len(kernel_sizes) == \
               len(n_channels) == \
               len(strides) == \
               len(paddings)
        super().__init__()

        self.hidden_sizes = hidden_sizes
        self.input_width = input_width
        self.input_height = input_height
        self.input_channels = input_channels
        self.output_size = output_size
        self.output_activation = output_activation
        self.hidden_activation = hidden_activation
        self.batch_norm_conv = batch_norm_conv
        self.batch_norm_fc = batch_norm_fc
        self.added_fc_input_size = added_fc_input_size
        self.conv_input_length = self.input_width * self.input_height * self.input_channels

        self.conv_layers = nn.ModuleList()
        self.conv_norm_layers = nn.ModuleList()
        self.fc_layers = nn.ModuleList()
        self.fc_norm_layers = nn.ModuleList()

        for out_channels, kernel_size, stride, padding in \
                zip(n_channels, kernel_sizes, strides, paddings):
            conv = nn.Conv2d(input_channels,
                             out_channels,
                             kernel_size,
                             stride=stride,
                             padding=padding)
            hidden_init(conv.weight)
            conv.bias.data.fill_(0)

            conv_layer = conv
            self.conv_layers.append(conv_layer)
            input_channels = out_channels

        # find output dim of conv_layers by trial and add normalization conv layers
        test_mat = torch.zeros(1, self.input_channels, self.input_width,
                               self.input_height)  # initially the model is on CPU (caller should then move it to GPU if
        for conv_layer in self.conv_layers:
            test_mat = conv_layer(test_mat)
            # import pdb; pdb.set_trace()
            self.conv_norm_layers.append(nn.BatchNorm2d(test_mat.shape[1]))

        fc_input_size = int(np.prod(test_mat.shape))
        # import pdb; pdb.set_trace()
        # used only for injecting input directly into fc layers
        fc_input_size += added_fc_input_size

        for idx, hidden_size in enumerate(hidden_sizes):
            fc_layer = nn.Linear(fc_input_size, hidden_size)

            norm_layer = nn.BatchNorm1d(hidden_size)
            fc_layer.weight.data.uniform_(-init_w, init_w)
            fc_layer.bias.data.uniform_(-init_w, init_w)

            self.fc_layers.append(fc_layer)
            self.fc_norm_layers.append(norm_layer)
            fc_input_size = hidden_size

        self.last_fc = nn.Linear(fc_input_size, output_size)
        self.last_fc.weight.data.uniform_(-init_w, init_w)
        self.last_fc.bias.data.uniform_(-init_w, init_w)

    def forward(self, input): # input is a batch of flattened images.
        h = input
        h = self.apply_forward(h, self.conv_layers, self.conv_norm_layers,
                               use_batch_norm=self.batch_norm_conv)
        # flatten channels for fc layers
        h = h.reshape(h.size(0), -1)
        h = self.apply_forward(h, self.fc_layers, self.fc_norm_layers,
                               use_batch_norm=self.batch_norm_fc)

        output = self.output_activation(self.last_fc(h))
        return output

    def apply_forward(self, input, hidden_layers, norm_layers,
                      use_batch_norm=False):
        h = input
        for layer, norm_layer in zip(hidden_layers, norm_layers):
            h = layer(h)
            if use_batch_norm:
                h = norm_layer(h)
            h = self.hidden_activation(h)
        return h



def gen_image_net(image_height, image_width, 
                  conv_kernel_sizes=[5, 3, 3 ,3], 
                  conv_n_channels=[16, 32, 64, 128], 
                  conv_strides=[3, 2, 2, 2]):
    conv_args=dict( # conv layers
        kernel_sizes=conv_kernel_sizes, # for sweep into, cartpole, drawer open. 
        n_channels=conv_n_channels,
        strides=conv_strides,
        output_size=1,
    )
    conv_kwargs=dict(
        hidden_sizes=[], # linear layers after conv
        batch_norm_conv=False,
        batch_norm_fc=False,
    )

    return CNN(
        **conv_args,
        paddings=np.zeros(len(conv_args['kernel_sizes']), dtype=np.int64),
        input_height=image_height,
        input_width=image_width,
        input_channels=3,
        init_w=1e-3,
        hidden_init=fanin_init,
        **conv_kwargs
    )

from PIL import Image
data_dir = 'preferences/filtered_state_images_CartPole-v1'
image_dir = data_dir + '/state_images'
image_path = image_dir + '/00001_000.png'
image = Image.open(image_path).convert("RGB")

w, h = image.size
model = gen_image_net(image_height=h, image_width=w)

df = pd.read_csv(data_dir + '/state_image_data.csv')

def generate_random_pairs(input_list, num_pairs):
    pairs = []
    for _ in range(num_pairs):
        if len(input_list) < 2:
            raise ValueError("The input list should have at least two distinct items")
        pair = random.sample(input_list, 2)
        pairs.append(pair)
    return pairs

my_list = ['a', 'b', 'c', 'd', 'e']
# num_pairs = 10
# print(generate_random_pairs(my_list, num_pairs))

def generate_all_pairs(input_list):
    pairs = []
    for i in range(len(input_list)):
        for j in range(len(input_list)):
            if i != j:
                pairs.append((input_list[i], input_list[j]))
    return pairs

my_list = ['a', 'b', 'c', 'd']
# print(generate_all_pairs(my_list))

def pair_up_randomly(lst):
    pairs = []
    n = len(lst)
    i = 0
    while i < n:
        if i == n-1:
            pairs.append([lst[i], lst[0]])
        else:
            pairs.append([lst[i], lst[i+1]])
        i += 2
    return pairs
my_list = ['a', 'b', 'c', 'd', 'e']
# print(pair_up_randomly(my_list))

import os
import torch
from torchvision import transforms
class ImageDataset(torch.utils.data.Dataset):
    def __init__(self,image_dir,file_df,transform=None):
        self.df = file_df
        self.image_dir = image_dir
#         self.pair_list = pair_up_randomly(list(range(0, len(self.df)))) # 99
#         self.pair_list = generate_random_pairs(list(range(0, len(self.df))), 5000)
        self.pair_list = generate_all_pairs(list(range(0, len(df))))
        self.transform = transform

    #dataset length
    def __len__(self):
        self.filelength = len(self.pair_list)
        return self.filelength

    #load an one of images
    def __getitem__(self,idx):
        img_name_1 = self.df.loc[self.pair_list[idx][0], 'filename']
        img_name_2 = self.df.loc[self.pair_list[idx][1], 'filename']
        img_path_1 = os.path.join(os.getcwd(), self.image_dir, img_name_1)
        img_path_2 = os.path.join(os.getcwd(), self.image_dir, img_name_2)
        img1 = transforms.ToTensor()(Image.open(img_path_1).convert('RGB'))
        img2 = transforms.ToTensor()(Image.open(img_path_2).convert('RGB'))
        r1 = self.df.loc[self.pair_list[idx][0], 'curated_reward']
        r2 = self.df.loc[self.pair_list[idx][1], 'curated_reward']

        if r1 >= r2:
          label = 0
        else:
          label = 1

        return img1, img2, torch.tensor(label)

batch_size = 32
epochs = 1
lr = 2e-5
dataset = ImageDataset(image_dir=image_dir, file_df = df)
dataloader = torch.utils.data.DataLoader(dataset = dataset, batch_size=batch_size, shuffle=True )

import sys
from tqdm import tqdm
from torch import nn
import torch
device = 'cuda' if torch.cuda.is_available() else 'cpu'

epoch_losses = []
epoch_accs   = []
model = model.to(device)
model.train()
opt = torch.optim.Adam(model.parameters(), lr = lr)
for epoch in range(1, epochs+1):
  losses = []
  accs = []
  for itr, (img1, img2, label) in tqdm(enumerate(dataloader), total=len(dataloader)):

    r1 = model(img1.to(device))
    r2 = model(img2.to(device))
    label = label.to(device)
    pred = torch.cat([r1, r2], dim=1)
#     print(f"{pred = }")
#     sys.exit()
    loss = nn.CrossEntropyLoss()(pred, label)

    cur_loss = loss.item()
    loss.backward()
    opt.step()
    opt.zero_grad()
    losses.append(cur_loss)

    # Calculate accuracy
    _, predicted = torch.max(pred.data, 1)
    predicted = predicted.to(device)
    total = label.size(0)
    correct = (predicted == label).sum().item()
    acc = correct / total
    accs.append(acc)
    
    if (itr + 1) % 100 == 0:
        print(f"Epoch {epoch}: Avg. Loss = {sum(losses) / len(losses) :<20}   Avg. Accuracy = {sum(accs) / len(accs) :<20}   {itr = }")

  # Calculate average loss and accuracy for the epoch
  avg_loss = sum(losses) / len(losses)
  avg_acc = sum(accs) / len(accs)

  print(f"Epoch {epoch}: Average Loss = {avg_loss}, Average Accuracy = {avg_acc}")

  epoch_losses.append(avg_loss)
  epoch_accs.append(avg_acc)
  torch.save(model.state_dict(), f"reward_model_curated_prefs_tanh.pth")
  print(f"{pred = }")
