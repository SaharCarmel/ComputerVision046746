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
transform = transforms.Compose([transforms.ToTensor()])
svhnData = torchvision.datasets.SVHN('data', split='train', transform=None,
target_transform=transform, download=True)
dataloader = torch.utils.data.DataLoader(svhnData,
                                          batch_size=4,
                                          shuffle=True,
                                          num_workers=0)
inputs, classes = next(iter(dataloader))
# %%
fig, axes = plt.subplots(1, 5, figsize=(12,2.5))
for i in data_loader:
    axes[i].imshow(svhnData[i][0])
    axes[i].set_title(svhnData[i][1])

# %%
