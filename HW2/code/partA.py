###############################################################################
#                            pkg imports                                      #
###############################################################################
import torch
import torchvision
import torch.nn             as nn
import torch.nn.functional  as F
import torch.optim          as optim
import os 
import matplotlib.pyplot    as plt
import scipy
import numpy                as np


###############################################################################
#                              classes                                        #
###############################################################################
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
    
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

###############################################################################
#                            functions                                        #
###############################################################################
def display_data(ds, batch_sz):
    dl = torch.utils.data.DataLoader(
        dataset = ds,
        batch_size = batch_sz,
        shuffle = False
    )
    iter_dl = iter(dl)
    images, _ = iter_dl.next()
    img = torchvision.utils.make_grid(images)
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


###############################################################################
#                                main                                         #
###############################################################################
def main():
    # define svhn download dir
    root = os.path.dirname(__file__)
    fp_svhn_dir = os.path.join(root, 'SVHN')
    
    # load dataset and download mat file
    ds_svhn = torchvision.datasets.SVHN(
        fp_svhn_dir,
        split = 'train',
        transform = torchvision.transforms.ToTensor(), #check the None requirement
        target_transform = None,
        download = False #change to true later
    )
    
    # display 5 images
    # display_data(ds_svhn, 5)
    batch_sz = 64
    dl_svhn = torch.utils.data.DataLoader(
        dataset = ds_svhn,
        batch_size = batch_sz,
        shuffle = False
    )

    dataiter = iter(dl_svhn)
    images, labels = dataiter.next()

    net = Net()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        net.parameters(), 
        lr = 0.001, 
        momentum = 0.9
    )
    
    for epoch in range(100):
        running_loss = 0.0
        for i, data in enumerate(dl_svhn, 0):
            inputs, labels = data
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 50 == 49:
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 50))
                running_loss = 0.0    
    
    print('finished training')

    PATH = os.path.join(fp_svhn_dir, 'net.pth')
    torch.save(net.state_dict(), PATH)

    correct = 0
    total = 0
    with torch.no_grad():
        for data in dl_svhn:
            images, labels = data
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the 10000 test images: %d %%' % (
        100 * correct / total))
    
if __name__ == "__main__":
    main()