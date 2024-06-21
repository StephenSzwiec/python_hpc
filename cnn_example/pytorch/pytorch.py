"""
CCAST Machine Learning Tutorial
Part 2: PyTorch 
Author: Stephen Szwiec
Date: 2023-12-29

This file contains the code for the PyTorch example of
a simple neural network. The code is based on the PyTorch
tutorial found at https://pytorch.org/tutorials/beginner/basics/
and uses the CIFAR-10 dataset, which can be found at 
https://www.cs.toronto.edu/~kriz/cifar.html.
"""

"""
Region: import libraries
"""
import os
import time
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils, datasets

"""
Region: global variables
"""
# initialize training parameters 
learning_rate = 0.001 # learning rate for gradient descent
epochs = 5 # number of iterations for gradient descent
batch = 500 # number of samples per batch 
display_step = 20 # how often to display training progress

# dataset parameters 
num_classes = 10 # number of classes in the dataset 

"""
Region: class definitions
"""
class NeuralNetwork(nn.Module):
    # define the network architecture
    # each network layer has a 3x3 kernel, stride of 1, and padding of 1
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            # Convolution Layer with 64 filters 
            nn.Conv2d(3, 32, 3, 1, 1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, 1, 1), nn.ReLU(),
            # Max Pooling Layer with 2x2 kernel and stride of 2
            nn.MaxPool2d(2, 2),
            # Convolution Layer with 128 filters
            nn.Conv2d(64, 128, 3, 1, 1), nn.ReLU(),
            nn.Conv2d(128, 128, 3, 1, 1), nn.ReLU(),
            # Max Pooling Layer with 2x2 kernel and stride of 2
            nn.MaxPool2d(2, 2),
            # Convolution Layer with 256 filters
            nn.Conv2d(128, 256, 3, 1, 1), nn.ReLU(),
            nn.Conv2d(256, 256, 3, 1, 1), nn.ReLU(),
            # Max Pooling Layer with 2x2 kernel and stride of 2
            nn.MaxPool2d(2, 2),
            # Flatten the data to a 1-D vector for the fully connected layer
            nn.Flatten(),
            # Fully Connected Layer with 1024 neurons
            nn.Linear(256*4*4, 1024), nn.ReLU(),
            # Dropout Layer with 25% dropout rate 
            nn.Dropout(0.25),
            # Fully Connected Layer with 10 neurons
            nn.Linear(1024, num_classes) 
        )

    # define the forward pass
    def forward(self, x):
        return self.network(x)

"""
Region: function definitions
"""
def get_default_device():
    if torch.cuda.is_available():
        print("Using CUDA")
        return "cuda" 
    else:
        print("Using CPU")
        return "cpu"

# utility function to neatly print times in SI units
def print_time(time):
    if time < 1e-6:
        print(f"{time*1e9:.5f} ns")
    elif time < 1e-3:
        print(f"{time*1e6:.5f} us")
    elif time < 1:
        print(f"{time*1e3:.5f} ms")
    else:
        print(f"{time:.5f} s")

# function to load CIFAR-10 dataset 
def load_cifar10(batch_size=128, device="cpu"):
    val_size = 10000 # number of samples in the validation set
    # define the data transformation 
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))])
    # load the training and test data to directory "data"
    training_data = datasets.CIFAR10( root = "data", train=True, download=True, transform=transform )
    val_data, training_data = torch.utils.data.random_split(training_data, [val_size, len(training_data)-val_size])
    test_data = datasets.CIFAR10( root = "data", train=False, download=True, transform=transform )
    # prepare the data for training, validation, and testing
    train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_data, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True)
    return train_dataloader, val_dataloader, test_dataloader

# accuracy metric function is sparse categorical accuracy weighted by class frequency
def score(pred, labels):
    return torch.sum(torch.argmax(pred, dim=1) == labels).item() / len(labels)

# training function
def train(dataloader, model, loss_fn, optimizer, device, display_step=20):
    num_batches = len(dataloader)
    scores = []
    start = time.time()
    model.train()
    for batch, (x, y) in enumerate(dataloader):
        # move the data to the device
        x, y = x.to(device), y.to(device)
        # compute the prediction and loss
        pred = model(x)
        loss = loss_fn(pred, y)
        # backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # update the accuracy
        scores.append(score(pred, y))
        # display the training progress
        if batch % (num_batches//display_step) == 0 and batch > 0:
            # print loss and accuracy
            current = batch 
            print(f"Current loss: {loss:>7f}  [{current:>5d}/{num_batches:>5d}]")
            print(f"Average accuracy: {100*torch.mean(torch.tensor(scores)):.5f}%")
            # print mean step time 
            step_time = (time.time() - start) / (num_batches//display_step)
            print(f"Time/step: ", end="")
            print_time(step_time)
            # reset the start step_time and scores
            start = time.time()
            scores = []

# testing function 
def test(dataloader, model, loss_fn, device):
    num_batches = len(dataloader)
    model.eval()
    test_loss = 0.0 
    scores = []
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            test_loss += loss_fn(pred, y).item()
            scores.append(score(pred, y))
    test_loss /= num_batches
    test_score = torch.mean(torch.tensor(scores))
    print(f"Test Loss: {test_loss:.7f}, Test accuracy: {100*test_score:.5f}%")

"""
Region: main program
"""
def main():
    # set up the device
    dtype = torch.float
    device = get_default_device() 
    # load the datasets
    train_set, val_set, test_set = load_cifar10(batch)
    # define the model and move it to the device
    model = NeuralNetwork().to(device)
    print(model)
    # define loss function and optimizer
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    start = time.time()
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train(train_set, model, loss_fn, optimizer, device)
        test(val_set, model, loss_fn, device)
    end = time.time()
    print(f"Training time:")
    print_time(end-start)
    # test model on validation set
    print("Final Test Accuracy:")
    test(test_set, model, loss_fn, device)

if __name__ == "__main__":
    main()
