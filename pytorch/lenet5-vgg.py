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
from tqdm import tqdm_notebook as tqdm

# Data visualization
import matplotlib.pyplot as plt
import numpy as np

#from google.colab import drive
#drive.mount('/content/drive')
# Change the current working directory to the Google drive folder for path to dataloader
#os.chdir('/content/drive/MyDrive/...')

# You cannot change this line.
from dataloader import CIFAR10


'''
Helper functions
'''
#-------------------------------------------------
def plot_tran_val_stats(train_ll, val_ll):

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

#-------------------------------------------------

"""
Code for loading checkpoint and recover epoch id.
"""
def load_checkpoint():
    CKPT_PATH = "./saved_model/model.h5"
    def get_checkpoint(ckpt_path):
        try:
            ckpt = torch.load(ckpt_path)
        except Exception as e:
            print(e)
            return None
        return ckpt

    ckpt = get_checkpoint(CKPT_PATH)
    if ckpt is None or TRAIN_FROM_SCRATCH:
        if not TRAIN_FROM_SCRATCH:
            print("Checkpoint not found.")
        print("Training from scratch ...")
        start_epoch = 0
        current_learning_rate = INITIAL_LR
    else:
        print("Successfully loaded checkpoint: %s" %CKPT_PATH)
        net.load_state_dict(ckpt['net'])
        start_epoch = ckpt['epoch'] + 1
        current_learning_rate = ckpt['lr']
        print("Starting from epoch %d " %start_epoch)

    print("Starting from learning rate %f:" %current_learning_rate)

#-------------------------------------------------


#-------------------------------------------------
# Inferencing Code - Dataloading support from specific Dropbox is broken:

def inference_test():
    from dataloader2 import CIFAR10_2
    import csv
    from PIL import Image

    transform_test = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize(mean_t, std_t),
             ]
    )

    CKPT_PATH = "./saved_model/ass5_adam_3x16__0.01_0.9_0.001_128_45.h5"
    PRED_CSV_PATH = "./data/predictions.csv"

    i_model = LeNet5()

    if device =='cpu':
      ckpt = torch.load(CKPT_PATH, map_location=torch.device('cpu'))
    else:
      ckpt = get_checkpoint(CKPT_PATH)

    i_model.load_state_dict(ckpt['net'])
    i_model.eval()

    i_model = i_model.to(device)


    i_data = np.load(os.path.join(DATAROOT, "cifar10-batches-images-test/cifar10-batches-images-test.npy"))

    #testset = torchvision.datasets.ImageFolder("./data/cifar10-batches-images-test", transform_test)
    #testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False, num_workers=1)

    #testset = CIFAR10_2(root=DATAROOT, train=False, download=True, transform=transform_test)
    ##testloader = torch.utils.data.DataLoader(testset, batch_size=VAL_BATCH_SIZE, shuffle=False, num_workers=1)
    #testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False, num_workers=1)

    i_optimizer = torch.optim.Adam(i_model.parameters(), lr=INITIAL_LR,
                                 weight_decay=REG)

    i_preds = []
    id = 0

    with open(PRED_CSV_PATH, 'w') as f:
      write = csv.writer(f)

      write.writerow(["Id", "Category"])

      # Disable gradient during validation
      with torch.no_grad():
          #for batch_id, (inputs, labels) in enumerate(testloader):
          for img in i_data:

            img = Image.fromarray(img)
            img = transform_test(img)

            # Copy inputs to device
            #inputs = inputs.to(device)
            img = img.to(device)

            ## Zero the gradient
            #optimizer.zero_grad()      # TODO Do I need?
            # Generate output from the DNN.
            #outputs = i_model(inputs)
            outputs = i_model(img.unsqueeze(0))

            write.writerow([id, outputs.argmax().item()])

            #id+=len(inputs)
            id+=1

#-------------------------------------------------

"""
Building the model
"""
# Create the neural network module: LeNet-5
class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        self.flatten = nn.Flatten() #dim=0 is maintained
        self.mod_arch = nn.Sequential(
#Brute force; BIGGERR (like VGG but half)
#83.8% accuracy 
            nn.Conv2d(3, 64, 3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(128, 256, 3),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 512, 3),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(512*5*5, 220),
            nn.BatchNorm1d(220),
            nn.ReLU(),
            nn.Linear(220, 90),
            nn.BatchNorm1d(90),
            nn.ReLU(),
            nn.Linear(90, 10),
        )

    def forward(self, x):
      logits = self.mod_arch(x)
      #return nn.Softmax(logits, dim=1)
      return logits

#-------------------------------------------------
"""
Hyperparameters
"""

# Setting some hyperparameters
TRAIN_BATCH_SIZE = 128
VAL_BATCH_SIZE = 100
INITIAL_LR = 0.01
MOMENTUM = 0.9
REG = 1e-3 #1e-5
EPOCHS = 60
DATAROOT = "./data"
CHECKPOINT_PATH = "./saved_model"

# FLAG for loading the pretrained model
TRAIN_FROM_SCRATCH = True

#-------------------------------------------------
"""
Data preprocessing + augmentation 

Normalization reference value for mean/std:
mean(RGB-format): (0.4914, 0.4822, 0.4465)
std(RGB-format): (0.2023, 0.1994, 0.2010)
"""

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


# Dataset Loader
trainset = CIFAR10(root=DATAROOT, train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=TRAIN_BATCH_SIZE, shuffle=True, num_workers=1)
valset = CIFAR10(root=DATAROOT, train=False, download=True, transform=transform_val)
valloader = torch.utils.data.DataLoader(valset, batch_size=VAL_BATCH_SIZE, shuffle=True, num_workers=1)

#-------------------------------------------------
# Specifying the device for computation
device = 'cuda' if torch.cuda.is_available() else 'cpu'
if device =='cuda':
    print("Training on GPU...")
else:
    print("Training on CPU...")

#-------------------------------------------------


net = LeNet5()

# Initializing the weights of Conv, FC layers with xavier initialization
#torch.nn.init.xavier_normal_(net.mod_arch[0].weight)
#torch.nn.init.xavier_normal_(net.mod_arch[4].weight)
#torch.nn.init.xavier_normal_(net.mod_arch[9].weight)
#torch.nn.init.xavier_normal_(net.mod_arch[12].weight)
#torch.nn.init.xavier_normal_(net.mod_arch[15].weight)

#torch.nn.init.xavier_uniform_(net.mod_arch[0].weight)
#torch.nn.init.xavier_uniform_(net.mod_arch[4].weight)
#torch.nn.init.xavier_uniform_(net.mod_arch[9].weight)
#torch.nn.init.xavier_uniform_(net.mod_arch[12].weight)
#torch.nn.init.xavier_uniform_(net.mod_arch[15].weight)

#Removed because no improvement

net = net.to(device)


# Load checkpoint
load_checkpoint()

#-------------------------------------------------
"""
In the targeted classification task, cross entropy loss with L2
regularization is used as the learning object.
"""

# Create loss function and specify regularization
criterion = nn.CrossEntropyLoss()

# Optimizer configuration
# L2 regularization through weight_decay
#optimizer = torch.optim.SGD(LeNet5_model.parameters(), lr=l_rate)
#optimizer = torch.optim.SGD(net.parameters(), lr=INITIAL_LR,
#                            weight_decay=REG, momentum=MOMENTUM)
#                            #momentum=momen)
optimizer = torch.optim.Adam(net.parameters(), lr=INITIAL_LR,
                             weight_decay=REG)
#optimizer = torch.optim.Adam(LeNet5_model.parameters(), lr=l_rate) #0.001


"""
Training and validating model over entire CIFAR-10 training dataset. 
"""

def training():
    # Switch to train mode
    net.train()

    #print("Training...")
    total_examples = 0
    correct_examples = 0

    train_loss = 0
    train_acc = 0
    # Train the training dataset for 1 epoch.

    total_examples = len(trainloader.dataset)

    for batch_idx, (inputs, targets) in enumerate(trainloader):
        # Copy inputs to device
        inputs = inputs.to(device)
        targets = targets.to(device)

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
        #_, predicted =
        # Calculate accuracy
        #total_examples += len(targets)  # set before for loop
        correct_examples += (outputs.argmax(1) == targets).type(torch.float).sum().item()
        #train_loss += loss
        train_loss += loss.item()


        #global_step += 1
        #if global_step % 100 == 0:
        #    avg_loss = train_loss / (batch_idx + 1)
        #pass
    avg_loss = train_loss / len(trainloader)

    avg_acc = correct_examples / total_examples
    print(f"Training loss: {avg_loss:>0.4f}, Training accuracy: {100*avg_acc:>0.1f}")
    #print("Training loss: %.4f, Training accuracy: %.4f" %(avg_loss, avg_acc))
    train_ll.append(avg_loss)
    train_al.append(avg_acc*100)
    print(f"\n{datetime.datetime.now()}")

def validation():
    # Switch to evalution mode
    net.eval()

    # Validate on the validation dataset
    #print("Validation...")
    #total_examples = 0
    total_examples = len(valloader.dataset)
    correct_examples = 0


    val_loss = 0
    val_acc = 0
    # Disable gradient during validation
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(valloader):
            # Copy inputs to device
            inputs = inputs.to(device)
            targets = targets.to(device)
            # Zero the gradient
            optimizer.zero_grad()
            # Generate output from the DNN.
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            # Calculate predicted labels
            #_, predicted = outputs.max(1)
            # Calculate accuracy
            #total_examples += len(targets)   # set at begining of validation section
            #correct_examples += len(predicted)
            correct_examples += (outputs.argmax(1) == targets).type(torch.float).sum().item()
            #val_loss += loss
            val_loss += loss.item()

    avg_loss = val_loss / len(valloader)
    avg_acc = correct_examples / total_examples
    #print("Validation loss: %.4f, Validation accuracy: %.4f" % (avg_loss, avg_acc))
    print(f"Validation loss: {avg_loss:>0.4f}, Validation accuracy: {100*avg_acc:>0.1f}")
    val_ll.append(avg_loss)
    val_al.append(avg_acc*100)





# To use for plotting
train_ll = []   # [train_loss_avg, ]
val_ll = []     # [val_loss_avg, ]
train_al = []   # [train_accuracy, ]
val_al = []     # [val_accuracy, ]


# Start the training/validation process
# The process should take about 5 minutes on a GTX 1070-Ti
# if the code is written efficiently.
global_step = 0
best_val_acc = 0

t0 = time.time()

for i in range(start_epoch, EPOCHS):
    print("\n\nEpoch %d:" %i)
    print(f"{datetime.datetime.now()}")

    training()

    validation()


    """
    Assignment 4(b)
    Learning rate is an important hyperparameter to tune. Specify a
    learning rate decay policy and apply it in your training process.
    Briefly describe its impact on the learning curveduring your
    training process.
    Reference learning rate schedule:
    decay 0.98 for every 2 epochs. You may tune this parameter but
    minimal gain will be achieved.
    Assignment 4(c)
    As we can see from above, hyperparameter optimization is critical
    to obtain a good performance of DNN models. Try to fine-tune the
    model to over 70% accuracy. You may also increase the number of
    epochs to up to 100 during the process. Briefly describe what you
    have tried to improve the performance of the LeNet-5 model.
    """
    DECAY_EPOCHS = 2
    DECAY = 0.80
    if i % DECAY_EPOCHS == 0 and i != 0:
        current_learning_rate *= DECAY
        for param_group in optimizer.param_groups:
            # Assign the learning rate parameter
            param_group['lr'] = current_learning_rate

        print("Current learning rate has decayed to %f" %current_learning_rate)

    # Save for checkpoint
    if avg_acc > best_val_acc:
        best_val_acc = avg_acc
        if not os.path.exists(CHECKPOINT_PATH):
            os.makedirs(CHECKPOINT_PATH)
        print("Saving ...")
        state = {'net': net.state_dict(),
                 'epoch': i,
                 'lr': current_learning_rate}
        torch.save(state, os.path.join(CHECKPOINT_PATH, 'model.h5'))

t1 = time.time()
totalt = t1-t0

print("\nOptimization finished.")
print(f"\nTotal Training and Testing Time: {totalt}")

plot_train_val_stats(train_ll, val_ll)


