import cv2
import matplotlib.pyplot as plt
import torch
import os
from cv_hw3_q1 import get_palette, predict_segment, load_images



# model constants
DEEP_LAB_CLASSES    = 21
DIR_DATA            = 'data'
SUBDIR_sahar        = 'sahar'
root = os.path.dirname(__file__)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

cap = cv2.VideoCapture('/home/sahar/Programming/ComputerVision046746/HW3/code/data/sahar.mp4')

for i in range(100):
    ret, frame = cap.read()
    if i == 1 or i == 50:
        plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        plt.savefig('data/sahar/frame{}.png'.format(i))
        # plt.show()

cap.release()
cv2.destroyAllWindows()

#%%

model = torch.hub.load('pytorch/vision:v0.5.0', 'deeplabv3_resnet101', pretrained = True)
model.eval()
model = model.to(device)
deeplab_palette = get_palette(DEEP_LAB_CLASSES)
fp_sahar = os.path.join(root, DIR_DATA, SUBDIR_sahar)
img_sahar = load_images(fp_sahar)
pred_rnd = predict_segment(model, img_sahar)
x = 2