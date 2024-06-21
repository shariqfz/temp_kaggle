from PIL import Image
from torchvision import transforms
import os
import sys
import torch
from torch import nn as nn
import numpy as np

def process_rlvlmf_frames(img):

    def split_image(img):
        width, height = img.size
        midpoint = width // 2

        # Split the image into two parts
        img1 = img.crop((0, 0, midpoint, height))
        img2 = img.crop((midpoint, 0, width, height))
        return img1, img2
    return split_image(img)

def prepocess_image(image):
    # print(f"before crop size = {image.size}")
    # Crop the image
    crop_amount = 200
    left = crop_amount
    top = 150 # 150
    right = image.width - crop_amount
    bottom = image.height - 50 # 50
    image = image.crop((left, top, right, bottom))

    # print(f"after crop size = {image.size}")
    

    image = image.resize((256, 256))
    preprocess = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    image = preprocess(image)
    return image

def load_image(image_path, env_name):

    image = Image.open(image_path).convert("RGB") 
    # print(f"{image_path = }")

    # Crop the image
    # img1 = image
    # crop_amount = 150
    # left = crop_amount
    # top = crop_amount
    # right = img1.width - crop_amount
    # bottom = img1.height - 60
    # img1 = img1.crop((left, top, right, bottom))

    preprocess = transforms.Compose([
        transforms.Resize(224),
        # ResizeShorterSide(256),
        # transforms.CenterCrop(224),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.Normalize((0.5,), (0.5,))
    ])

    if 'rlvlmf' in env_name:
        img1, img2 = process_rlvlmf_frames(image)
        img1 = img1.resize((256, 256))
        img2 = img2.resize((256, 256))

        img1 = preprocess(img1)
        img2 = preprocess(img2)

        return img1, img2

    # image = resize_image(image, (256, 256))
    image = image.resize((256, 256))
    # transforms.Resize(224)(image).save(f"./annotated_preference_images_pairs/temp_img_{image_path.split('/')[-2]}.png")
    # sys.exit()
    image = preprocess(image)
    return image

def predict_reward(image, reward_model):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n{device = }")

    env_name_0 = "rlvlmf_CartPole" # args.env_name.split("-")[0]
    reward_models_dir = "/raid/infolab/veerendra/shariq/workdir/trained_reward_models/"
    model_path = os.path.join(reward_models_dir, f"reward_model_{env_name_0}.pth")
    h, w = 224, 224
    reward_model = gen_image_net(image_height=h, image_width=w)
    reward_model.load_state_dict(torch.load(model_path))
    reward_model.eval()
    print("loaded reward model")
    print(reward_model)
    input_image_path = "/raid/infolab/veerendra/shariq/workdir/frames_CartPole/seed_0_0.png"
    img1, img2 = load_image(input_image_path, env_name_0)
    print(f"{img1.shape = }")
    print(f"{img2.shape = }")
    with torch.no_grad():
        r_hat_1 = reward_model(img1.unsqueeze(0))
        r_hat_2 = reward_model(img2.unsqueeze(0))

    print(f"{r_hat_1 = }")
    print(f"{r_hat_2 = }")
