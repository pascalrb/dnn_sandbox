import time

import torch
import torchvision.transforms as transforms
import torchvision
import torch.nn as nn
import torch.optim as optim

from pruned_layers import *

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def train(net, epochs=100, batch_size=128, lr=0.01, reg=5e-4):
    """
    Training a network
    :param net:
    :param epochs:
    :param batch_size:
    """
    print('==> Preparing data..')
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),

    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),

    ])
    best_acc = 0  # best test accuracy
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=16)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=reg, nesterov=False)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[int(epochs*0.5), int(epochs*0.75)], gamma=0.1)

    global_steps = 0
    start = time.time()

    for epoch in range(start_epoch, epochs):
        """
        Start the training code.
        """
        print('\nEpoch: %d' % epoch)
        net.train()
        train_loss = 0
        correct = 0
        total = 0
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            loss.backward()

            optimizer.step()
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            global_steps += 1

            if global_steps % 16 == 0:
                end = time.time()
                num_examples_per_second = 16 * batch_size / (end - start)
                print("[Step=%d]\tLoss=%.4f\tacc=%.4f\t%.1f examples/second"
                      % (global_steps, train_loss / (batch_idx + 1), (correct / total)*100, num_examples_per_second))
                start = time.time()

        scheduler.step()

        """
        Start the testing code.
        """
        net.eval()
        test_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(testloader):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = net(inputs)
                loss = criterion(outputs, targets)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        num_val_steps = len(testloader)
        val_acc = correct / total
        print("Test Loss=%.4f, Test acc=%.4f" % (test_loss / (num_val_steps), val_acc*100))

        if val_acc > best_acc:
            best_acc = val_acc
            print("Saving...")
            torch.save(net.state_dict(), "net_before_pruning.pt")



def finetune_after_prune(net, epochs=100, batch_size=128, lr=0.01, reg=5e-4):
    print('==> Preparing data..')
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),

    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),

    ])
    best_acc = 0  # best test accuracy
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=16)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

    net = net.to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=reg, nesterov=False)

    global_steps = 0
    start = time.time()

    for epoch in range(start_epoch, epochs):
        """
        Start the training code.
        """
        print('\nEpoch: %d' % epoch)
        net.train()
        train_loss = 0
        correct = 0
        total = 0
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            # TODO: If quantized, need to handle decoding weight indices before forward pass
            outputs = net(inputs)   
            loss = criterion(outputs, targets)
            loss.backward()

            # before optimizer.step(), manipulate the gradient of pruned weights
            for n, m in net.named_modules():
                if isinstance(m, PrunedConv):
                    m.conv.weight.grad.data = m.conv.weight.grad.to(device) * m.mask.to(device)
                elif isinstance(m, PruneLinear):
                    m.linear.weight.grad.data = m.linear.weight.grad.to(device) * m.mask.to(device)

            optimizer.step()
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            global_steps += 1

            if global_steps % 16 == 0:
                end = time.time()
                num_examples_per_second = 16 * batch_size / (end - start)
                print("[Step=%d]\tLoss=%.4f\tacc=%.4f\t%.1f examples/second"
                      % (global_steps, train_loss / (batch_idx + 1), (correct / total)*100, num_examples_per_second))
                start = time.time()
            #TODO: fine-tuning after quantization, for non-zero masks:
            #           aggregate the gradients in the same patterns (clusters)
            #           as the weights
            #           THEN, mul with lr, subtract it from current centroids
            #                   Then update the codebook with that 

        """
        Start the testing code.
        """
        net.eval()
        test_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(testloader):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = net(inputs)
                loss = criterion(outputs, targets)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        num_val_steps = len(testloader)
        val_acc = correct / total
        print("Test Loss=%.4f, Test acc=%.4f" % (test_loss / (num_val_steps), val_acc*100))

        if val_acc > best_acc:
            best_acc = val_acc
            print("Saving...")
            torch.save(net.state_dict(), "net_after_pruning.pt")

def test(net, is_quantized=False, centers=None):
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),

    ])

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    criterion = nn.CrossEntropyLoss()

    net.eval()
    test_loss = 0
    correct = 0
    total = 0

    # Decode the quantized weights from codebook if the weights have been quantized
    lay_idx = 0
    quant_weights = []
    if(is_quantized):
        for n, m in net.named_modules():
            if isinstance(m, PrunedConv):
                # Saving quantized weight to requantize after forward testing pass
                # because huffman_coding() expect quantized weights
                quant_weights.append(m.conv.weight.detach())

                centroids = centers[lay_idx]
                weight = m.conv.weight.flatten().detach().numpy().astype(int)
                m.conv.weight.data = torch.tensor(centroids[weight], dtype=m.conv.weight.dtype).reshape_as(m.conv.weight).to(device)
                m.conv.weight.data = m.conv.weight.to(device) * m.mask.to(device)
                lay_idx += 1
            elif isinstance(m, PruneLinear):
                # Saving quantized weight to requantize after forward testing pass
                # because huffman_coding() expect quantized weights
                quant_weights.append(m.linear.weight.detach())

                centroids = centers[lay_idx]
                weight = m.linear.weight.flatten().detach().numpy().astype(int)
                m.linear.weight.data = torch.tensor(centroids[weight], dtype=m.linear.weight.dtype).reshape_as(m.linear.weight).to(device)
                m.linear.weight.data = m.linear.weight.to(device) * m.mask.to(device)
                lay_idx += 1

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()


    # Reset the weights back to centroid indices becasue huffman_coding function expects a quantized model
    lay_idx = 0   
    if(is_quantized):
        for n, m in net.named_modules():
            if isinstance(m, PrunedConv):
                m.conv.weight.data = quant_weights[lay_idx] 
                m.conv.weight.data = m.conv.weight.to(device) * m.mask.to(device)
                lay_idx += 1
            elif isinstance(m, PruneLinear):
                m.linear.weight.data = quant_weights[lay_idx] 
                m.linear.weight.data = m.linear.weight.to(device) * m.mask.to(device)
                lay_idx += 1

            
    num_val_steps = len(testloader)
    val_acc = correct / total
    print("Test Loss=%.4f, Test accuracy=%.4f" % (test_loss / (num_val_steps), val_acc*100))



