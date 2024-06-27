import os
import sys
import csv
import torch
from torch import nn as nn
import numpy as np
import tyro
from dataclasses import dataclass

from catdogmodel import CatAndDogConvNet
from RlvlmfCNNmodel import gen_image_net
from utils_train_reward_model import load_image, process_rlvlmf_frames

from torch.utils.tensorboard import SummaryWriter
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

from PIL import Image
from torchvision import transforms
import warnings
warnings.filterwarnings("ignore")

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"\n{device = }")

@dataclass
class Args:
    lr: float = 2e-5
    """learning rate"""
    epochs: int = 50
    batch_size: int = 8
    env_name: str = 'rlvlmf_CartPole'
    data_path: str = f'./preferences/{env_name[7:]}_pref.pkl'
    seed: int = 42
    image_dir: str = f'frames_{env_name[7:]}'

args = tyro.cli(Args)
seed= args.seed
# random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

lr = args.lr
num_epochs = args.epochs
batch_size = args.batch_size

os.system(f"mkdir -p runs/{args.env_name}")
writer = SummaryWriter(f"runs/{args.env_name}")
writer.add_text(
    "hyperparameters",
    "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
)




# import torchvision.transforms as transforms

# import math
# from torchvision.transforms import functional as F

# class ResizeShorterSide:
#     def __init__(self, size):
#         self.size = size

#     def __call__(self, img):
#         w, h = img.size
#         if w < h:
#             new_w = self.size
#             new_h = int(self.size * h / w)
#         else:
#             new_h = self.size
#             new_w = int(self.size * w / h)
#         return F.resize(img, (new_h, new_w))

# Usage


# img = Image.open("image.jpg")
# transform = transforms.Compose([ResizeShorterSide(256)])
# out_img = transform(img)


# cartpole-vo: image dims= (400, 600, 3)

# image_path1 = "./frames/image_globalstep_0002_envstep_0001_envid_0.png"
# image_path2 = "./frames/image_globalstep_4992_envstep_0127_envid_0.png"
# imgt1 = load_image(image_path1)
# imgt2 = load_image(image_path2)
# ch, h, w = imgt1.shape
# print(f"{ch, h, w}")
h, w = 224, 224
model = gen_image_net(image_height=h, image_width=w).to(device)
# Calculate the number of parameters
num_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f'Rlvlmf model num paramteres =  {num_parameters}')

# model = CatAndDogConvNet().to(device)
# num_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f'cat dog model has {num_parameters} parameters.')
# sys.exit()

from torch.utils.data import Dataset


class ImageDataset(Dataset):
    def __init__(self, image_paths, image_dir):
        self.image_paths = image_paths
        self.image_dir = image_dir
        # self.labels = labels

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):

        if 'rlvlmf' in args.env_name:
            image_path = self.image_paths[idx][0]
            image_path = os.path.join(os.getcwd(), self.image_dir, image_path)
            image1, image2 = load_image(image_path, args.env_name)
            label = self.image_paths[idx][-1]

            # print(f"{image1.shape = } {image2.shape =}")
            # from torchvision.transforms import ToPILImage
            # img_tmp = image1.cpu()
            # pil_image = ToPILImage()(img_tmp)
            # pil_image.save("pil_image_train_reward.png", 'png')
            # sys.exit()
            return image1, image2, label

        image_path_1 = self.image_paths[idx][0]
        image_path_2 = self.image_paths[idx][1]
        image_path_1 = os.path.join(os.getcwd(), self.image_dir, image_path_1)
        image_path_2 = os.path.join(os.getcwd(), self.image_dir, image_path_2)
        image1 = load_image(image_path_1)
        image2 = load_image(image_path_2)
        label = self.image_paths[idx][-1]

        print(f"{image1.shape = } {image2.shape =}")
        sys.exit()

        # print(f"{idx = }  {self.image_paths[idx] = }")
        # print(f"{label = }\n{image1.shape = }\n{image1.shape = }")
        # sys.exit()
        return image1, image2, label


total = 0
paramlst = model.parameters()
opt = torch.optim.Adam(paramlst, lr = lr)

import pickle

# Open the pickle file and load the list
# with open(args.data_path, 'rb') as f:
#     preferences = pickle.load(f)

with open('/raid/infolab/veerendra/shariq/workdir/preferences/sim_pref_frames_CartPole.csv', 'r') as file:
    reader = csv.reader(file)
    sims = []       # 0.90837,seed_0_665.png,1
    for row in reader:
        sims.append(row)
n = len(sims)
print(f"{sims[:2] = }")
preferences = [[x[1], int(x[2])] for x in sims[: n // 2]] 

preferences_filtered = [x for x in preferences if x[-1] != -1]
print(f"\n\n{len(preferences_filtered) = }") 
print(f"{len(preferences) = }    {preferences[0] = }\n") 

print(f"preferences path : {args.data_path}")
print(f"{args.env_name = }")

from torch.utils.data import DataLoader
from tqdm import tqdm
import sys

dataset = ImageDataset(image_paths=preferences_filtered, image_dir=args.image_dir)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

losses = []
acc = 0
model.train()
global_step = 0
for epoch in range(num_epochs):
    
    epoch_loss = 0.0
        
    for i, (sa_t_1, sa_t_2, labels) in tqdm(enumerate(dataloader), total=len(dataloader)):

        sa_t_1 = sa_t_1.to(device)
        sa_t_2 = sa_t_2.to(device)
        labels = labels.long().to(device)
        
        total += labels.size(0)
    
        # if self.image_reward:
            # sa_t_1 is batch_size x segment x image_height x image_width x 3
            # sa_t_1 = sa_t_1.transpose(2, 3).transpose(1, 2) # for torch we need to transpose channel first
            # sa_t_2 = sa_t_2.transpose(2, 3).transpose(1, 2) # for torch we need to transpose channel first

        # get logits
        r_hat1 = model(sa_t_1)
        r_hat2 = model(sa_t_2)

        r_hat = torch.cat([r_hat1, r_hat2], axis=-1)
        # print(f"{r_hat.shape = } {labels.shape = }")
        # print(f"{r_hat = } {labels = }")
        
        # compute loss
        curr_loss = nn.CrossEntropyLoss()(r_hat, labels)
        epoch_loss += curr_loss.item()
        # loss = curr_loss
        losses.append(curr_loss.item())
        
        opt.zero_grad()
        curr_loss.backward()
        opt.step()

        ###################################################
        # logits = model(sa_t_1)
        # curr_loss = nn.CrossEntropyLoss()(logits, labels)
        # epoch_loss += curr_loss.item()

        # # opt.zero_grad()
        # # curr_loss.backward()
        # # opt.step()

        # max_value, predicted = torch.max(logits.data, 1)
        # correct = (predicted == labels).sum().item()
        # acc += correct
        # # acc = acc / total
        # # print(f"{max_value = } {predicted = }  {logits = } {labels = }")

        # logits = model(sa_t_2)
        # curr_loss += nn.CrossEntropyLoss()(logits, labels)
        # epoch_loss += curr_loss.item()

        # opt.zero_grad()
        # curr_loss.backward()
        # opt.step()

        # max_value, predicted = torch.max(logits.data, 1)
        # correct = (predicted == labels).sum().item()
        # acc += correct
        # acc = acc / (total * 2)
        # print(f"{max_value = } {predicted = }  {logits = }")


        # print(f"epoch = {epoch+1}\tloss = {curr_loss.item()}\tacc = {acc}") if (epoch+1) % 1 == 0 else None
        # sys.exit()

        ###################################################

        # compute acc
        max_value, predicted = torch.max(r_hat.data, 1)
        correct = (predicted == labels).sum().item()
        acc += correct

        # print(f"{max_value = } {predicted = }  {r_hat = }")
        # print(f"{sa_t_1 = } {sa_t_2 = }  ")
        # sys.exit()
   
        if (global_step + 1) % 100:
            # log the running loss
            writer.add_scalar('train_loss',
                            curr_loss.item(),
                            global_step)

        global_step += 1

    epoch_loss = epoch_loss / len(dataloader)
    acc = acc / (args.batch_size * len(dataloader))
    print(f"epoch = {epoch+1}\t{epoch_loss = }\tacc = {acc}") if (epoch+1) % 1 == 0 else None
    epoch_loss, acc = 0.0, 0.0
    
writer.close()
# torch.cuda.empty_cache()


# save model
torch.save(model.state_dict(), f"./trained_reward_models/reward_model_catanddog_{args.env_name}.pth")

# load model
# model = TheModelClass(*args, **kwargs)
# model.load_state_dict(torch.load('model_path.pth'))