# # : Comment / What the function does
# """""" : Function call code

# Library import
import os
import torch
import torchvision
import tarfile
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torchvision.datasets.utils import download_url
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torchvision.transforms as tt
from torch.utils.data import random_split
from torchvision.utils import make_grid
from torchvision.utils import make_grid
import matplotlib
import matplotlib.pyplot as plt
import IPython
import numpy
import numpy as np
import seaborn as sns
import pandas as pd
import tarfile




# We can look at batches of images from the dataset using the `make_grid` method from `torchvision`.
#  Each time the following code is run, we get a different bach, since the sampler shuffles the indices before creating batches. 
"""
show_batch(train_dl)
"""
def show_batch(dl):
    for images, labels in dl:
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.set_xticks([]); ax.set_yticks([])
        ax.imshow(make_grid(images, nrow=16).permute(1, 2, 0))
        break

# Accuracy Plot using matplolib (validation set)
"""
plot_accuracies(history)
"""
def plot_accuracies(history):
    accuracies = [x['val_acc'] for x in history]
    plt.plot(accuracies, '-x')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.title('Accuracy vs. No. of epochs');


# Loss Plot using matplotlib (validation set)
"""
plot_losses(history)
"""
def plot_losses(history):
    train_losses = [x.get('train_loss') for x in history]
    val_losses = [x['val_loss'] for x in history]
    plt.plot(train_losses, '-bx')
    plt.plot(val_losses, '-rx')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(['Training', 'Validation'])
    plt.title('Loss vs. No. of epochs');


# Learning rate change with time / batch by batch plot using matplotlib (validation set)
"""
plot_lrs(history)
"""
def plot_lrs(history):
    lrs = np.concatenate([x.get('lrs', []) for x in history])
    plt.plot(lrs)
    plt.xlabel('Batch no.')
    plt.ylabel('Learning rate')
    plt.title('Learning Rate vs. Batch no.');


# Evalutes model and validation set 
"""
history = [evaluate(model, valid_dl)]
history
"""
def evaluate(model, val_loader):
    model.eval()
    outputs = [model.validation_step(batch) for batch in val_loader]
    return model.validation_epoch_end(outputs)


# Cuda check for Pytorch
"""
cuda_check()
"""
def cuda_check():
    # setting device on GPU if available, else CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)
    print()

    #Additional Info when using cuda
    if device.type == 'cuda':
        print('Cuda Available: ', torch.cuda.is_available())
        print('Device Name: ' ,torch.cuda.get_device_name(0))
        print('Memory Usage:')
        print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
        print('Cached:   ', round(torch.cuda.memory_cached(0)/1024**3,1), 'GB')
        print('Index: ',torch.cuda.current_device())
        print('Device Location: ', torch.cuda.device(0))
        print('Number of GPUs Available: ', torch.cuda.device_count())


def get_default_device():
    """Pick GPU if available, else CPU"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')
    
def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list,tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)

class DeviceDataLoader():
    """Wrap a dataloader to move data to a device"""
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device
        
    def __iter__(self):
        """Yield a batch of data after moving it to device"""
        for b in self.dl: 
            yield to_device(b, self.device)

    def __len__(self):
        """Number of batches"""
        return len(self.dl)


def ver_check():
    # print('Library / Module Versions -')
    print('Pytorch Version: ' , torch.__version__)
    print('Matplotlib Version: ' , matplotlib.__version__)
    print('Numpy Version: ', numpy.__version__)
    print('Seaborn Version: ', sns.__version__)
    print('Pandas Version: ', pd.__version__)
    print('IPython System Info: ', IPython.sys_info())


# import tarfile
def tarr(file_name):
    my_tar = tarfile.open(file_name)
    # file_name_2 = 
    my_tar.extractall('./my_folder') # specify which folder to extract to
    print("Extracted to ./my_folder")
    my_tar.close()

# print current path / working directory
def path_check():
    kroll = os.path.abspath(os.getcwd())
    print(kroll)

# def env_check():
#     print('Pip: ')
#     !pip list # type: ignore
#     print('Conda env list: ')
#     !conda env list # type: ignore
#     print('Conda list: ')
#     !conda list # type: ignore
