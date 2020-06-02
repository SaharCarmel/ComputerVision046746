# %%
import cv2
import matplotlib.pyplot as plt
import torch
import os
from cv_hw3_q1 import get_palette, predict_segment, load_images,  paste_image, plot_prediction
from PIL import Image, ImageFilter
import numpy as np
from frame_video_convert import video_to_image_seq, image_seq_to_video
from matplotlib import cm

def saveToFolder(vidPath):
    filename = os.path.basename(vidPath).split('.')[0]
    datafolder = os.path.join(os.getcwd(), 'data', filename)
    if not os.path.exists(datafolder):
        print('Data for {} does not exists, creating new folder'.format(filename))    
        video_to_image_seq(vidPath, datafolder)
    return datafolder


def binary_mask(img, mask, colors_palette):
    img_mask = Image.fromarray(mask.byte().cpu().numpy()).resize(img.size)
    img_mask.putpalette(colors_palette)
    bw_mask = img_mask.convert('L')
    return bw_mask

class backgroundIter():
    def __init__(self, folderPath):
        self.folderPath = folderPath
        self.filesList = os.listdir(folderPath)
        self.count = -1
    def __iter__(self):
        return self

    def __next__(self):
        self.count += 1
        _imagePath = os.path.join(self.folderPath, self.filesList[self.count]) 
        return Image.open(_imagePath)


def extractGreen(img):
    empty_img = np.zeros_like(img)
    RED, GREEN, BLUE = (2, 1, 0)

    reds = img[:, :, RED]
    greens = img[:, :, GREEN]
    blues = img[:, :, BLUE]
    mask = (greens < 35) | (reds > greens) | (blues > greens)
    result = np.where(mask, 255, 0)
    return result

folderPaths = []
vidPath1= os.path.join(os.getcwd(),'data', 'sahar.mp4')
vidPath2= os.path.join(os.getcwd(),'data', 'blast.mp4')
videoList = [vidPath1, vidPath2]

for videoPath in videoList:
    print('Working on {}'.format(videoPath))
    folderPaths.append(saveToFolder(videoPath))


#%%
# model constants
DEEP_LAB_CLASSES = 21
DIR_DATA = 'data'
SUBDIR_sahar = 'sahar'
save_path = os.path.join(os.getcwd(), 'data', 'to_vid')
root = os.path.dirname(__file__)
device = torch.device(
    'cuda:0' if torch.cuda.is_available() else
    'cpu')

model = torch.hub.load('pytorch/vision:v0.5.0',
                       'deeplabv3_resnet101', pretrained=True)
model.eval()
model = model.to(device)
deeplab_palette = get_palette(DEEP_LAB_CLASSES)
img_sahar = load_images(folderPaths[0], rot=0)
pred_rnd = predict_segment(model, img_sahar)
effect_images = load_images(folderPaths[1])
# bgLoader = backgroundIter(folderPaths[1])
bg = Image.open(os.path.join(os.getcwd(), 'data', 'pokemon.jpg'))
lu = [256 for x in range(256)]
lu[0] = 0
lu_green = np.zeros(np.array(effect_images[0]).shape[0:2])


# %%
import shutil
shutil.rmtree(os.path.join(os.getcwd(),'data', 'to_vid'))
os.makedirs(os.path.join(os.getcwd(),'data', 'to_vid'))
filter_func = lambda x: 0 if x ==0 else 256
from matplotlib import cm
for i, (image, effect) in enumerate(zip(pred_rnd, effect_images)):
    image, img_mask = image
    effect, effect_mask = effect, extractGreen(np.array(effect))
    binary_mask_sahar = binary_mask(image, img_mask, deeplab_palette)
    binary_mask_sahar = binary_mask_sahar.point(lu)
    binary_mask_effect = binary_mask(effect, torch.tensor(effect_mask), deeplab_palette)
    binary_mask_effect = binary_mask_effect.point(lu)
    # bg = next(bgLoader)
    bgcopy = bg.copy()
    # plt.imshow(binary_mask_effect)
    # plt.show()
    # image.paste(effect, (0,0), binary_mask_effect)
    # plt.imshow(image)
    # plt.show()
    bgcopy = bg.resize(image.size)
    sahar_edit = paste_image(
        front=image,
        bg=bgcopy,
        os = (0,0),
        mask=binary_mask_sahar,
        resize=False
    )
    sahar_edit = paste_image(
        front = effect,
        bg=sahar_edit,
        os = (0,0),
        mask=binary_mask_effect,
        resize=False
    )

    sahar_edit.save(os.path.join(os.getcwd(), 'data', 'to_vid', "{}.jpg".format(i)))
    # _, axes = plt.subplots(1, 1)
    # axes.imshow(sahar_edit)
    # axes.set_xticks([])
    # axes.set_yticks([])
    # plt.show()
#%%

savePath = os.path.join(os.getcwd(),'data', 'to_vid')
image_seq_to_video(savePath, fps=30)


# %%
