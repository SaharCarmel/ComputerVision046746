#%% 
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import wandb
import os
import numpy as np
import time
print(os.getcwd())




# wandb.init(project="SVHN-project")
#%% config
# WandB – Initialize a new run
# wandb.init(entity="carmel", project="SVHN-project")

# WandB – Config is a variable that holds and saves hyperparameters and inputs

hyperparameter_defaults = dict(
    # dropout = 0.5,
    # channels_one = 16,
    # channels_two = 32,
    batch_size = 128,
    lr = 0.001,
    epochs = 2,
    )
wandb.init(config=hyperparameter_defaults)
config = wandb.config
classses = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '0']
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
criterion = nn.CrossEntropyLoss()
#%% Network
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Conv Layer block 2
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=0.05),

            # Conv Layer block 3
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten()
        )

        self.linear = nn.Sequential(
            nn.Dropout(p=0.1),
            nn.Linear(4096, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(512, 10)
        )
    
    def forward(self, x):
        x = self.conv(x)
        x = self.linear(x)
        return x
#%% 

def convert_to_imshow_format(image):
    # first convert back to [0,1] range from [-1,1] range - approximately...
    image = image / 2 + 0.5
    image = image.numpy()
    # convert from CHW to HWC
    # from 3x32x32 to 32x32x3
    return image.transpose(1,2,0)

def calculate_accuracy(model, dataloader, device, classes,step,  set_='train'):
    model.eval() # put in evaluation mode
    total_correct = 0
    total_images = 0
    confusion_matrix = np.zeros([10,10], int)
    with torch.no_grad():
        for data in dataloader:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total_images += labels.size(0)
            total_correct += (predicted == labels).sum().item()
            for i, l in enumerate(labels):
                confusion_matrix[l.item(), predicted[i].item()] += 1 
    model_accuracy = total_correct / total_images * 100
    if set_ == 'test':
        example_images = []
        example_images.append(wandb.Image(
            data[0], caption="Pred: {} Truth: {}".format(classes[predicted[0].item()], classes[labels[0]])))
        wandb.log({
            "Examples": example_images,
            "Test Accuracy": model_accuracy}, step=step)
    return model_accuracy, confusion_matrix
#%% dataloading
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

# Normalize the test set same as training set without augmentation
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
trainset = torchvision.datasets.SVHN('SVHN', split='train', transform=transform_train,
target_transform=None, download=False)

testset = torchvision.datasets.SVHN('SVHN', split='test', transform=transform_test,
target_transform=None, download=True)
trainLoader = torch.utils.data.DataLoader(trainset,
                                          batch_size=5,
                                          shuffle=True,
                                          num_workers=0)

testLoader = torch.utils.data.DataLoader(testset,
                                          batch_size=config.batch_size,
                                          shuffle=True,
                                          num_workers=0)

#%%
images, classes = next(iter(trainLoader))
fig, axes = plt.subplots(1, 5, figsize=(12,2.5))
for i, (image, class_) in enumerate(zip(images, classes)):
    axes[i].imshow(convert_to_imshow_format(image))
    axes[i].set_title(np.array(class_))

# %%
model = Net().to(device)
pytorch_total_params = sum(p.numel() for p in model.parameters())

#%%
# Magic
wandb.watch(model, log="all")

optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

for epoch in range(1, config.epochs + 1):
    model.train()  # put in training mode
    running_loss = 0.0
    epoch_time = time.time()
    for i, data in enumerate(trainLoader, 0):
        # get the inputs
        inputs, labels = data
        # send them to device
        inputs = inputs.to(device)
        labels = labels.to(device)

        # forward + backward + optimize
        outputs = model(inputs)  # forward pass
        loss = criterion(outputs, labels)  # calculate the loss
        # always the same 3 steps
        optimizer.zero_grad()  # zero the parameter gradients
        loss.backward()  # backpropagation
        optimizer.step()  # update parameters

        # print statistics
        running_loss += loss.data.item()

    # Normalizing the loss by the total number of train batches
    running_loss /= len(trainLoader)

    # Calculate training/test set accuracy of the existing model
    train_accuracy, _ = calculate_accuracy(model, trainLoader, device,classses,i, set_='train')
    test_accuracy, _ = calculate_accuracy(model, testLoader, device,classses,i, set_='test')
    wandb.log({"Train Accuracy":train_accuracy, "Epoch":i}, step=i)

    log = "Epoch: {} | Loss: {:.4f} | Training accuracy: {:.3f}% | Test accuracy: {:.3f}% | ".format(epoch, running_loss, train_accuracy, test_accuracy)
    epoch_time = time.time() - epoch_time
    log += "Epoch Time: {:.2f} secs".format(epoch_time)
    print(log)
    
    # save model
    if epoch % 20 == 0:
        print('==> Saving model ...')
        if not os.path.isdir('checkpoints'):
            os.mkdir('checkpoints')
        torch.save(model.state_dict(), './checkpoints/SVHN.h5')
        wandb.save('model.h5')

print('==> Finished Training ...')

# %%
