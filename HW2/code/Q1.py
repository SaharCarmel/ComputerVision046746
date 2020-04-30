#%% 
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
#%% defs

#%% Network
class Net(nn.module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv = nn.Sequential(
            
        )
        self.linear = nn.Sequential(

        )
#%% dataloading
transform = transforms.Compose([
    transforms.ToTensor()
])

# dataset and data loader
svhn_dataset = torchvision.datasets.SVHN(root='data',
                                split='train',
                                transform=transform,
                                download=True)
inputs, classes = next(iter(svhn_dataset))
# %%
fig, axes = plt.subplots(1, 5, figsize=(12,2.5))
for i, (image, class_) in enumerate(zip(inputs, classes)):
    axes[i].imshow(image)
    axes[i].set_title(class_)

# %%
