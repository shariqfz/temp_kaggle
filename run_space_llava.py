import os
import csv
import random
import numpy as np
import pandas as pd
import torch
import sys
sys.path.append(os.getcwd())
os.environ["CUDA_VISIBLE_DEVICES"] = "5"

orig_stdout = sys.stdout
orig_stderr = sys.stderr
sys.stdout = orig_stdout
sys.stderr = orig_stderr

seed=42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
from utils_train_reward_model import load_image

import io
import base64
from PIL import Image

from llama_cpp import Llama
from llama_cpp.llama_chat_format import Llava15ChatHandler


env_name = "ClothFold"

def image_to_base64_data_uri(image_input):
    # Check if the input is a file path (string)
    if isinstance(image_input, str):
        with open(image_input, "rb") as img_file:
            base64_data = base64.b64encode(img_file.read()).decode('utf-8')

    # Check if the input is a PIL Image
    elif isinstance(image_input, Image.Image):
        buffer = io.BytesIO()
        image_input.save(buffer, format="PNG")  # You can change the format if needed
        base64_data = base64.b64encode(buffer.getvalue()).decode('utf-8')

    else:
        raise ValueError("Unsupported input type. Input must be a file path or a PIL.Image.Image instance.")

    return f"data:image/png;base64,{base64_data}"

mmproj= "/raid/infolab/veerendra/shariq/spacellava/mmproj-model-f16.gguf"
model_path= "/raid/infolab/veerendra/shariq/spacellava/ggml-model-q4_0.gguf"
chat_handler = Llava15ChatHandler(clip_model_path=mmproj, verbose=True)
spacellava = Llama(model_path=model_path, chat_handler=chat_handler, n_ctx=2048, logits_all=True, n_gpu_layers = -1)


######################################################################################################

# CartPole prompt

# system_prompt = "You are given two images of a cartpole, and you have to give answer by comparing the two images with respect to the goal provided."
system_prompt = "You are given two images of a black cart and a brown pole."

# task_description = "A pole is attached by an un-actuated joint to a cart, which moves along a frictionless track. " \
#                   + "The pendulum is placed upright on the cart and the goal is to balance the pole by applying forces in the " \
#                   + "left and right direction on the cart."

# task_description = "Goal is to balance the brown pole on the black cart to be upright."

# prompt = f"{task_description} " \
#          + "Is the goal better achieved in image 1 or image 2? Reply a single line " \
#          + "of 0 if the goal is better achieved in image 1, or 1 if the goal is better " \
#          + "achieved in image 2. Reply -1 if the text is unsure or there is no difference."

# prompt = "Answer in only one word: right, left, or none if there is no difference in the images. In which of the two images (left, right, or none) the pole in more upright."
prompt = "Answer in only one word: right, left, or none. In which of the two images (left, right, or none) the pole in more upright."

######################################################################################################

# ClothFold prompt
system_prompt = "Your are given two images of a cloth."
prompt = "Answer in only one word: right, left, or none. In which of the two images (left, right, or none) the cloth is folded more diagonally."




import os
import sys

def get_preference(image_path_1, system_prompt, prompt):
  data_uri_1 = image_to_base64_data_uri(image_path_1)
  # data_uri_2 = image_to_base64_data_uri(image_path_2)
  messages = [
      {"role": "system", "content": system_prompt},
      {
          "role": "user",
          "content": [
              # {"type" : "text", "text": "Image 1:"},
              {"type": "image_url", "image_url": {"url": data_uri_1}},
              # {"type" : "text", "text": "Image 2:"},
              # {"type": "image_url", "image_url": {"url": data_uri_2}},
              {"type" : "text", "text": prompt},
              ]
          }
      ]
  original_stdout = sys.stdout
  original_stderr = sys.stderr
  sys.stdout = open(os.devnull, 'w')
  sys.stderr = open(os.devnull, 'w')
  results = spacellava.create_chat_completion(messages = messages)
  sys.stdout = original_stdout
  sys.stderr = original_stderr
  return results["choices"][0]["message"]["content"].strip()

# image_paths = os.listdir('./frames')
image_paths = os.listdir(f'./frames_{env_name}')
image_paths.sort()

sampled_images = image_paths

from tqdm import tqdm
import pickle
from PIL import Image
import matplotlib.pyplot as plt
os.system("mkdir -p annotated_preference_images_pairs")
def concatenate_images(image_path_1, image_path_2):
    img1 = Image.open(image_path_1)
    img2 = Image.open(image_path_2)

    # Crop the image
    crop_amount = 200
    left = crop_amount
    top = 150
    right = img1.width - crop_amount
    bottom = img1.height - 50
    img1 = img1.crop((left, top, right, bottom))
    img2 = img2.crop((left, top, right, bottom))
    
    # Concatenate the images
    width, height = img1.size
    new_img = Image.new('RGB', (width*2, height))
    new_img.paste(img1, (0, 0))
    new_img.paste(img2, (width, 0))

     # Create a figure with the label as title
    # plt.figure(figsize=(8,4))
    # plt.imshow(new_img)
    # plt.title(label)
    # plt.axis('off')

    # Save the figure
    new_img.save(f"./annotated_preference_images_pairs/test2_preferenced_image.png")
    # plt.savefig(f"./annotated_preference_images_pairs/test_preferenced_image.png", bbox_inches = 'tight',  pad_inches = 0.1)
    return new_img

from collections import defaultdict
map_label = defaultdict(lambda: -1, {"left":0, "Left": 0, "right": 1, "Right": 1})

preferences = []
raw_preferences = []

def generate_preference_labels(image_path_list, image_dir='frames'):

  items = image_path_list
  n = len(items)
  # Randomly select 1000 pairs
  # pairs = [random.sample(items, 2) for _ in range(10000)]
  pairs = items # rlvlmf
  print(f"{len(pairs) = }")

  for i, pair in tqdm(enumerate(pairs), total=len(pairs)):
      # print(i, end= ' ')
      # if (i+1) % 25 == 0:
      #   print()

      # image_path_1, image_path_2 = pair
      # preferences.append([image_path_1, image_path_2])
      # raw_preferences.append([image_path_1, image_path_2])
      # image_path_1 = os.path.join(os.getcwd(), image_dir, image_path_1)
      # image_path_2 = os.path.join(os.getcwd(), image_dir, image_path_2)

      # cont_img = concatenate_images(image_path_1, image_path_2)
      # image_path_1 = cont_img

      ## rlvlmf ##
      preferences.append([pair])
      raw_preferences.append([pair])
      print(f"{pair = }")
      image = Image.open(f'./frames_{env_name}/' + pair).convert("RGB") 
      image_path_1 = image
      image.save(f"./annotated_preference_images_pairs/test4_preferenced_image.png")
      ############

      # sys.exit()

      original_stdout = sys.stdout
      original_stderr = sys.stderr
      sys.stdout = open(os.devnull, 'w')
      sys.stderr = open(os.devnull, 'w')
      raw_preference = get_preference(image_path_1, system_prompt, prompt)
      sys.stdout = original_stdout
      sys.stderr = original_stderr
      # preference = -2
      int_preference = map_label[raw_preference]

      preferences[-1].append(int_preference)
      raw_preferences[-1].append(raw_preference)
      # preferences[-1].append(preference)

      if (i+1) % 10 == 0:
        with open(f'preferences_{env_name}_spacellava.pkl', 'wb') as f:
          pickle.dump(preferences, f)
        df = pd.DataFrame(raw_preferences)
        df.to_csv(f'raw_preferences_{env_name}.csv', header=False, index=False)

  with open(f'preferences_{env_name}_spacellava.pkl', 'wb') as f:
    pickle.dump(preferences, f)
  return preferences

with open(f'/raid/infolab/veerendra/shariq/workdir/preferences/{env_name}_pref.pkl', 'rb') as f:
  true_pref = pickle.load(f)


########### with similarity filtering #################
with open(f'/raid/infolab/veerendra/shariq/workdir/preferences/sim_pref_frames_{env_name}.csv', 'r') as file:
    reader = csv.reader(file)
    sims = []       # 0.90837,seed_0_665.png,1
    for row in reader:
        sims.append(row)

sampled_images = []
true_pref = []
for x in sims[: len(sims) // 2]:
  sampled_images.append(x[1])
  true_pref.append([x[1], int(x[-1])])

print(f"to perform preferencing on {len(sampled_images) = }")


#######################################################
preferences = generate_preference_labels(image_path_list=sampled_images, image_dir='frames')

print(f"{true_pref[:3] = }\t{len(true_pref) = }")
print(f"{preferences[:3] = }\t{len(preferences) = }")



labels = [x[-1] for x in preferences]

from collections import Counter
cntr = Counter(labels)
print(cntr)
print()

pred_pref = preferences
true_pref.sort(key = lambda x: x[0])
pred_pref.sort(key = lambda x: x[0])

cnt = 0
sum = 0
for x, y in zip(true_pref, pred_pref):
    sum += int(x[-1] == y[-1])
    cnt += 1

print(f"Acc : {sum/cnt}")


df = pd.DataFrame(raw_preferences)
df.to_csv(f'raw_preferences_{env_name}.csv', header=False, index=False)
# print(df.head())