import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import time

def get_mnist_loaders(batch_size=64, num_workers=2):
    """
    Create MNIST data loaders for training and testing
    """
    # Define transformations
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # MNIST mean and std
    ])
    
    # Download and load the training data
    train_dataset = datasets.MNIST(root='./data', 
                                  train=True, 
                                  download=True, 
                                  transform=transform)
    
    # Download and load the test data
    test_dataset = datasets.MNIST(root='./data', 
                                 train=False, 
                                 download=True, 
                                 transform=transform)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, 
                             batch_size=batch_size, 
                             shuffle=True, 
                             num_workers=num_workers)
    
    test_loader = DataLoader(test_dataset, 
                            batch_size=batch_size, 
                            shuffle=False, 
                            num_workers=num_workers)
    
    return train_loader, test_loader

def train_epoch(model, train_loader, criterion, optimizer, device):
    """
    Train the model for one epoch
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, target)
        
        # Backward and optimize
        loss.backward()
        optimizer.step()
        
        # Track statistics
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()
        
        # Print progress
        if (batch_idx + 1) % 100 == 0:
            print(f'Batch: {batch_idx+1}/{len(train_loader)}, Loss: {loss.item():.4f}, Acc: {100.*correct/total:.2f}%')
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100. * correct / total
    
    return epoch_loss, epoch_acc

def evaluate(model, test_loader, criterion, device):
    """
    Evaluate the model on the test set
    """
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            
            # Sum up batch loss
            test_loss += criterion(outputs, target).item()
            
            # Get the index of the max log-probability
            _, predicted = outputs.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
    
    test_loss /= len(test_loader)
    test_acc = 100. * correct / total
    
    return test_loss, test_acc

def plot_training_history(train_losses, train_accs, test_losses, test_accs):
    """
    Plot training and testing loss and accuracy curves
    """
    plt.figure(figsize=(12, 5))
    
    # Plot losses
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.plot(test_losses, label='Testing Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss Curves')
    
    # Plot accuracies
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Training Accuracy')
    plt.plot(test_accs, label='Testing Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.title('Accuracy Curves')
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.show()

def visualize_predictions(model, test_loader, device, num_samples=10):
    """
    Visualize model predictions on sample test images
    """
    model.eval()
    
    # Get a batch of test images
    dataiter = iter(test_loader)
    images, labels = next(dataiter)
    
    # Make predictions
    images = images[:num_samples].to(device)
    labels = labels[:num_samples]
    outputs = model(images)
    _, predicted = torch.max(outputs, 1)
    
    # Plot images with predictions
    fig = plt.figure(figsize=(15, 3))
    for i in range(num_samples):
        ax = fig.add_subplot(1, num_samples, i+1)
        img = images[i].cpu().numpy().squeeze()
        ax.imshow(img, cmap='gray')
        title = f"Pred: {predicted[i].item()}\nTrue: {labels[i].item()}"
        ax.set_title(title, color=("green" if predicted[i] == labels[i] else "red"))
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('predictions.png')
    plt.show()

def count_parameters(model):
    """
    Count the number of trainable parameters in the model
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
