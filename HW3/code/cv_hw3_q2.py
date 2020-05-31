# %%
import cv2
import matplotlib.pyplot as plt
import torch
import os
from cv_hw3_q1 import get_palette, predict_segment, load_images,  paste_image, binary_mask
from PIL import Image, ImageFilter
import numpy as np


# model constants
DEEP_LAB_CLASSES = 21
DIR_DATA = 'data'
SUBDIR_sahar = 'sahar2'
root = os.path.dirname(__file__)
device = torch.device(
    # 'cuda:0' if torch.cuda.is_available() else
    'cpu')
# %%

model = torch.hub.load('pytorch/vision:v0.5.0',
                       'deeplabv3_resnet101', pretrained=True)
model.eval()
model = model.to(device)
deeplab_palette = get_palette(DEEP_LAB_CLASSES)
fp_sahar = os.path.join(root, DIR_DATA, SUBDIR_sahar)
img_sahar = load_images(fp_sahar)
pred_rnd = predict_segment(model, img_sahar)
# %%
bgpath = os.path.join(os.getcwd(), 'data', 'green.jpg')
bg = Image.open(bgpath)
bg.thumbnail((102, 181))

#%%
def binary_mask(img, mask, colors_palette):
    img_mask = Image.fromarray(mask.byte().cpu().numpy()).resize(img.size)
    img_mask.putpalette(colors_palette)
    bw_mask = img_mask.convert('L')
    return bw_mask
# %%
for image in pred_rnd:
    image, img_mask = image
    binary_mask_sahar = binary_mask(image, img_mask, deeplab_palette)
    sahar_pokemon = paste_image(
        front=image,
        bg=bg,
        os = (300, 150),
        mask=binary_mask_sahar,
        resize=False
    )
    _, axes = plt.subplots(1, 1)
    axes.imshow(sahar_pokemon)
    axes.set_xticks([])
    axes.set_yticks([])
    plt.show()



# %%
