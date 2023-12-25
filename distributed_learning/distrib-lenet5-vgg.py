import argparse
import os, sys
import time
import datetime

import time

# Import pytorch dependencies
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torchvision.models as models
#from tqdm import tqdm_notebook as tqdm

# Data visualization
import matplotlib.pyplot as plt
import numpy as np

# Multiprocessing
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
import psutil

#from google.colab import drive
#drive.mount('/content/drive')
# Change the current working directory to the Google drive folder for path to dataloader
#os.chdir('/content/drive/MyDrive/...')

#sys.path.insert(0, './tools')
sys.path.insert(0, './../lenet5_vgg/tools')
#sys.path.insert(0, '/home/ubuntu/dnn_sandbox/pytorch/tools')

# You cannot change this line.
from dataloader import CIFAR10


"""
Hyperparameters/Global Constants
"""
TRAIN_BATCH_SIZE = 128
VAL_BATCH_SIZE = 100
INITIAL_LR = 0.01
MOMENTUM = 0.9
REG = 1e-3 #1e-5
EPOCHS = 30 #60
DATAROOT = "./data"
CHECKPOINT_PATH = "./saved_model"
DIRECT_CKPT_PATH = "./saved_model/model.h5"

# FLAG for loading the pretrained model
TRAIN_FROM_SCRATCH = True
CURRENT_LR = INITIAL_LR


'''
Visualization for traning and validation loss and accuracy
'''
def plot_train_val_stats(train_ll, val_ll, train_al, val_al):

    # Matplotlib data visualization
    plt.plot(train_ll)
    plt.plot(val_ll)
    plt.xlabel('Epoch')
    plt.ylabel('Avg Loss')
    plt.title(f"{optimizer.__class__.__name__}\nlr: {INITIAL_LR} | mom: {MOMENTUM} | l2reg: {REG} | t_bs: {TRAIN_BATCH_SIZE} | epoch: {EPOCHS}")
    ax = plt.gca()
    ax.legend(['Training Avg Loss', 'Eval Avg Loss'])
    for i in range(EPOCHS):
      if i % 2 == 0 or i == EPOCHS-1:
        #plt.text(i-1, train_ll[i-1], f"{(train_ll[i-1]):>0.2f}", fontsize='xx-small', ha="center")
        plt.text(i, train_ll[i], f"{(train_ll[i]):>0.2f}|{train_al[i]:>0.1f}%", fontsize='xx-small', ha="center")
        plt.text(i, val_ll[i], f"{(val_ll[i]):>0.2f}|{val_al[i]:>0.1f}%", fontsize='xx-small', ha="center")
    plt.show()



"""
Data preprocessing + augmentation 

Normalization reference value for mean/std:
mean(RGB-format): (0.4914, 0.4822, 0.4465)
std(RGB-format): (0.2023, 0.1994, 0.2010)
"""
def data_processing():

    mean_t = (0.4914, 0.4822, 0.4465)
    std_t  = (0.2023, 0.1994, 0.2010)

    # Specify preprocessing function.
    # Reference mean/std value for
    transform_train = transforms.Compose(
            [transforms.ToTensor(),
             # Applies normalization on the input
             transforms.Normalize(mean_t, std_t),
             transforms.RandomCrop(32),
             transforms.RandomHorizontalFlip(0.5),
             ]
        )

    transform_val = transforms.Compose(
            [transforms.ToTensor(),
             # Applies normalization on the input data
             transforms.Normalize(mean_t, std_t),
             transforms.RandomCrop(32),
             transforms.RandomHorizontalFlip(0.5),
             ]
        )


    # Pulling Dataset
    trainset = CIFAR10(root=DATAROOT, train=True, download=True, transform=transform_train)
    valset   = CIFAR10(root=DATAROOT, train=False, download=True, transform=transform_val)

    return trainset, valset



"""
Training for 1 epoch
"""
def training(device, net, optimizer, trainloader):
    # Switch to train mode
    net.train()

    # Create loss function and specify regularization
    criterion = nn.CrossEntropyLoss()
    # Optimizer configuration
    #optimizer = torch.optim.Adam(net.parameters(), lr=INITIAL_LR,
    #                            weight_decay=REG)

    #print("Training...")
    total_examples = 0
    correct_examples = 0

    train_loss = 0
    train_acc = 0

    # Train the training dataset for 1 epoch.
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        # Copy inputs to device
        inputs = inputs.to(device)
        targets = targets.to(device)

        total_examples += len(inputs)

        # Zero the gradient
        optimizer.zero_grad()
        # Generate output
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        # Now backward loss
        loss.backward()
        # Apply gradient
        optimizer.step()

        # Calculate predicted labels
        correct_examples += (outputs.argmax(1) == targets).type(torch.float).sum().item()
        train_loss += loss.item()


    avg_loss = train_loss / len(trainloader)

    print(f"TOTAL_EX SIZE: {total_examples} | TRAINLOADER SIZE (trainloader): {len(trainloader)}")
    avg_acc = correct_examples / total_examples
    print(f"PID: {os.getpid()} - Training loss: {avg_loss:>0.4f}, Training accuracy: {100*avg_acc:>0.1f}")
    print(f"{datetime.datetime.now()}")

    return avg_acc


"""
Validation/testing for 1 epoch
"""
def validation(device, net, valloader):
    # Switch to evalution mode
    net.eval()

    # Create loss function and specify regularization
    criterion = nn.CrossEntropyLoss()

    # Validate on the validation dataset
    #print("Validation...")
    total_examples = 0
    correct_examples = 0


    val_loss = 0
    # Disable gradient during validation
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(valloader):
            # Copy inputs to device
            inputs = inputs.to(device)
            targets = targets.to(device)

            total_examples += len(inputs)
            
            # Generate output from the DNN.
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            correct_examples += (outputs.argmax(1) == targets).type(torch.float).sum().item()
            val_loss += loss.item()

    avg_loss = val_loss / len(valloader)
    avg_acc = correct_examples / total_examples
    print(f"Validation loss: {avg_loss:>0.4f}, Validation accuracy: {100*avg_acc:>0.1f}")

    return avg_acc*100




"""
Training and validating model over entire CIFAR-10 training dataset. 
"""
def train_validate(device, net, trainloader, valloader, current_lr):
    #global CURRENT_LR

    # Optimizer configuration
    optimizer = torch.optim.Adam(net.parameters(), lr=INITIAL_LR,
                                weight_decay=REG)

    best_val_acc = 0

    # Tracking elapsed time for training and validating model
    t0 = time.time()

    pid = os.getpid()

    for i in range(EPOCHS):
        print(f'\n\nEpoch: {i} - PID: {pid}')
        print(f"{datetime.datetime.now()}")

        training(device, net, optimizer, trainloader)
        avg_acc = validation(device, net, valloader)
        
        """
        Learning rate decaying
        """
        DECAY_EPOCHS = 2
        DECAY = 0.80
        if i % DECAY_EPOCHS == 0 and i != 0:
            current_lr *= DECAY
            for param_group in optimizer.param_groups:
                # Assign the learning rate parameter
                param_group['lr'] = current_lr

            print("Current learning rate has decayed to %f" %current_lr)

        # Save for checkpoint
        if avg_acc > best_val_acc:
            best_val_acc = avg_acc
            if not os.path.exists(CHECKPOINT_PATH):
                os.makedirs(CHECKPOINT_PATH)
            print("Saving ...")
            torch.save(net.state_dict(), os.path.join(CHECKPOINT_PATH, 'model.h5'))



    t1 = time.time()
    totalt = t1-t0

    print("\nTraining finished.")
    print(f"\nTotal Training: {totalt/60:>0.2f}m{totalt%60:>0.2f}s")


#-------------------------------------------------

"""
Model base class
"""
# Create the neural network module: LeNet-5
class LeNet5_VGG(nn.Module):
    def __init__(self):
        super(LeNet5_VGG, self).__init__()
        self.flatten = nn.Flatten() #dim=0 is maintained
        self.mod_arch = nn.Sequential(
# Inital LeNet5 model shape (~60% accuracy)
# Now ~72.2% with 10% random dropout 
            nn.Conv2d(3, 6, 5),
            nn.BatchNorm2d(6),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(6, 16, 5),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Flatten(),             
            nn.Dropout(0.1),
            nn.Linear(16*5*5, 120),   
            nn.BatchNorm1d(120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.BatchNorm1d(84),
            nn.ReLU(),
            nn.Linear(84, 10),
#------------------------------------

##Brute force; BIGGERR (like VGG but half)
##83.8% accuracy 
#            nn.Conv2d(3, 64, 3),
#            nn.BatchNorm2d(64),
#            nn.ReLU(),
#            nn.Conv2d(64, 128, 3),
#            nn.BatchNorm2d(128),
#            nn.ReLU(),
#            nn.MaxPool2d(2, 2),
#
#            nn.Conv2d(128, 256, 3),
#            nn.BatchNorm2d(256),
#            nn.ReLU(),
#            nn.Conv2d(256, 512, 3),
#            nn.BatchNorm2d(512),
#            nn.ReLU(),
#            nn.MaxPool2d(2, 2),
#
#            nn.Flatten(),
#            nn.Dropout(0.5),
#            nn.Linear(512*5*5, 220),
#            nn.BatchNorm1d(220),
#            nn.ReLU(),
#            nn.Linear(220, 90),
#            nn.BatchNorm1d(90),
#            nn.ReLU(),
#            nn.Linear(90, 10),
        )

    def forward(self, x):
      logits = self.mod_arch(x)
      #return nn.Softmax(logits, dim=1)
      return logits



#########################################################################
#                                                                       #
#            "entrypoint" of distrib-lenet5-vgg.py module               #
#                                                                       #
#########################################################################

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='PyTorch Distributed Training')
    parser.add_argument('--auto_distrib', type=bool, default=False, help ="Automatic dataset distribution to workers")
    parser.add_argument('--num_workers', type=int, default=1, help ="Number of workers to distribute dataset to")

    args = parser.parse_args()

    #-------------------------------------------------
    # Specifying the device for computation
    if torch.cuda.is_available():
        device = 'cuda'
        print("Training on GPU...\n")
    #elif torch.backends.mps.is_available():
    #    device = torch.device("mps")
    #    print("Training on MPS...\n")
    else:
        device = 'cpu'
        print("Training on CPU...\n")
    #-------------------------------------------------

    net = LeNet5_VGG()
    #net = models.resnet18(pretrained=True)
    #net = models.alexnet(pretrained=True)
    #net = models.vgg16(pretrained=True)
    #net = models.densenet161(pretrained=True)

    #net = net.to(device, memory_format=torch.channels_last) #(slower with 6 procs on mac)
    net = net.to(device)
    

    # Multiprocessing 
    if args.auto_distrib:
        #num_of_cores = mp.cpu_count()
        num_of_cores = psutil.cpu_count(logical=False)
    else:
        num_of_cores = args.num_workers

    #num_of_cores = 6
    print(f'Number of cores: {num_of_cores}')
    num_of_procs = num_of_cores
    net.share_memory()
    processes = []
    trainset, valset = data_processing()

    for proc in range(num_of_procs):
        # Start the training/validation process
        trainloader = torch.utils.data.DataLoader(trainset, 
                                                    sampler=DistributedSampler(dataset=trainset,
                                                                                num_replicas=num_of_procs,
                                                                                rank=proc),
                                                    batch_size=TRAIN_BATCH_SIZE)#, num_workers=4)#, num_workers=1)
        valloader = torch.utils.data.DataLoader(valset,  
                                                    sampler=DistributedSampler(dataset=valset,
                                                                                num_replicas=num_of_procs,
                                                                                rank=proc),
                                                    batch_size=VAL_BATCH_SIZE)#, num_workers=4)#, num_workers=1)

        p = mp.Process(target=train_validate, args=(device, net, trainloader, valloader, CURRENT_LR))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    print(f'Final Validation:')
    #valloader = torch.utils.data.DataLoader(valset, batch_size=VAL_BATCH_SIZE, shuffle=True, num_workers=1)
    acc1 = validation(device, net, valloader)
    acc2 = validation(device, net, valloader)
    acc3 = validation(device, net, valloader)
    totalacc = (acc1 + acc2 + acc3)/3
    print(f'avg = {totalacc}')



    # Saving model in "PyTorch" format
    if not os.path.exists(CHECKPOINT_PATH):
        os.makedirs(CHECKPOINT_PATH)
    torch.save(net.state_dict(), os.path.join(CHECKPOINT_PATH, 'model.h5'))

    # Saving model in "TorchScript" format (for deployment)
    torch.jit.script(net).save(os.path.join(CHECKPOINT_PATH, 'model_TS.pt'))
