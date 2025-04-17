import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import time
import os
from models import MS_RAN
from utils import (
    get_mnist_loaders,
    train_epoch,
    evaluate,
    plot_training_history,
    visualize_predictions,
    count_parameters
)

def main():
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Hyperparameters
    batch_size = 128
    num_epochs = 10
    learning_rate = 0.001
    weight_decay = 1e-4

    # Create data loaders
    train_loader, test_loader = get_mnist_loaders(batch_size=batch_size)
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Testing samples: {len(test_loader.dataset)}")

    # Initialize the model
    model = MS_RAN().to(device)
    print(f"Model parameters: {count_parameters(model):,}")

    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.5, verbose=True)

    # Training loop
    train_losses = []
    train_accs = []
    test_losses = []
    test_accs = []
    best_acc = 0.0

    print("Starting training...")
    start_time = time.time()

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")

        # Train for one epoch
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        train_losses.append(train_loss)
        train_accs.append(train_acc)

        # Evaluate on test set
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        test_losses.append(test_loss)
        test_accs.append(test_acc)

        # Update learning rate
        scheduler.step(test_loss)

        # Print epoch statistics
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")

        # Save best model
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), 'best_model.pth')
            print(f"Model saved with accuracy: {best_acc:.2f}%")

    total_time = time.time() - start_time
    print(f"\nTraining completed in {total_time/60:.2f} minutes")
    print(f"Best test accuracy: {best_acc:.2f}%")

    # Plot training history
    plot_training_history(train_losses, train_accs, test_losses, test_accs)

    # Load best model and visualize predictions
    model.load_state_dict(torch.load('best_model.pth'))
    visualize_predictions(model, test_loader, device)

if __name__ == '__main__':
    main()
