import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import time
from tqdm import tqdm

from clean_pruned_model import CleanResNet

# PyTorch imports
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

# Add project root to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '../..'))
sys.path.insert(0, project_root)

def load_training_data(data_dir, batch_size=64, num_workers=4):
    """
    Load training and validation data from directory.
    
    Args:
        data_dir: Path to data directory (should contain train and val folders)
        batch_size: Batch size for data loaders
        num_workers: Number of workers for data loading
        
    Returns:
        tuple: (train_loader, val_loader)
    """
    # Define data transforms for training and validation
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Load datasets
    try:
        train_dir = os.path.join(data_dir, 'train')
        val_dir = os.path.join(data_dir, 'val')
        
        # Check if directory exists
        if not os.path.exists(train_dir):
            print(f"Warning: Training directory {train_dir} not found")
            # Fall back to using the main directory for both training and validation
            train_dir = data_dir
        
        if not os.path.exists(val_dir):
            print(f"Warning: Validation directory {val_dir} not found")
            # Fall back to using the main directory for validation
            val_dir = data_dir
        
        # Create datasets
        train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
        val_dataset = datasets.ImageFolder(val_dir, transform=val_transform)
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, 
            num_workers=num_workers, pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False, 
            num_workers=num_workers, pin_memory=True
        )
        
        print(f"Loaded {len(train_dataset)} training images and {len(val_dataset)} validation images")
        print(f"Found {len(train_dataset.classes)} classes")
        
        return train_loader, val_loader
    
    except Exception as e:
        print(f"Error loading training data: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def retrain_model(model, data_loaders, epochs=10, lr=0.001, momentum=0.9, weight_decay=1e-4, output_dir="retrained_model"):
    """
    Retrain a pruned model to recover accuracy.
    
    Args:
        model: PyTorch model to retrain
        data_loaders: Tuple of (train_loader, val_loader)
        epochs: Number of training epochs
        lr: Learning rate
        momentum: Momentum for SGD optimizer
        weight_decay: Weight decay for regularization
        output_dir: Directory to save model checkpoints and logs
        
    Returns:
        model: Retrained model
    """
    try:
        train_loader, val_loader = data_loaders
        if train_loader is None or val_loader is None:
            print("Error: No data loaders provided")
            return model
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Setup device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        
        # Move model to device
        model.to(device)
        
        # Define loss function and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
        
        # Training loop
        print("\n=== Starting Retraining ===")
        best_acc = 0.0
        history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
        
        start_time = time.time()
        for epoch in range(epochs):
            epoch_start = time.time()
            
            # Training phase
            model.train()
            running_loss = 0.0
            running_corrects = 0
            
            # Progress bar for training
            with tqdm(train_loader, unit="batch") as tepoch:
                tepoch.set_description(f"Epoch {epoch+1}/{epochs}")
                
                for inputs, labels in tepoch:
                    inputs, labels = inputs.to(device), labels.to(device)
                    
                    # Zero the parameter gradients
                    optimizer.zero_grad()
                    
                    # Forward pass
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    
                    # Backward pass and optimize
                    loss.backward()
                    optimizer.step()
                    
                    # Update statistics
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)
                    
                    # Update progress bar
                    tepoch.set_postfix(loss=loss.item())
            
            # Calculate epoch loss and accuracy
            epoch_train_loss = running_loss / len(train_loader.dataset)
            epoch_train_acc = running_corrects.double() / len(train_loader.dataset)
            
            # Validation phase
            model.eval()
            running_loss = 0.0
            running_corrects = 0
            
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)
            
            # Calculate validation loss and accuracy
            epoch_val_loss = running_loss / len(val_loader.dataset)
            epoch_val_acc = running_corrects.double() / len(val_loader.dataset)
            
            # Update learning rate
            scheduler.step()
            
            # Calculate epoch time
            epoch_time = time.time() - epoch_start
            
            # Print statistics
            print(f"Epoch {epoch+1}/{epochs} [{epoch_time:.1f}s] - "
                  f"Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.4f}, "
                  f"Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc:.4f}")
            
            # Save history
            history['train_loss'].append(epoch_train_loss)
            history['val_loss'].append(epoch_val_loss)
            history['train_acc'].append(epoch_train_acc.item())
            history['val_acc'].append(epoch_val_acc.item())
            
            # Save model if it's the best so far
            if epoch_val_acc > best_acc:
                best_acc = epoch_val_acc
                best_model_path = os.path.join(output_dir, 'best_model.pth')
                torch.save(model, best_model_path)
                print(f"New best model saved to {best_model_path}")
        
        # Save final model
        final_model_path = os.path.join(output_dir, 'final_model.pth')
        torch.save(model, final_model_path)
        print(f"Final model saved to {final_model_path}")
        
        # Save retrained model with standard name
        retrained_path = os.path.join(output_dir, 'retrained_model.pth')
        torch.save(model, retrained_path)
        print(f"Retrained model saved to {retrained_path}")
        
        # Plot training history
        plot_history(history, os.path.join(output_dir, 'training_history.png'))
        
        # Calculate total training time
        total_time = time.time() - start_time
        print(f"\nRetraining completed in {total_time:.1f} seconds!")
        print(f"Best validation accuracy: {best_acc:.4f}")
        
        # Return the best model
        best_model = torch.load(best_model_path)
        return best_model
        
    except Exception as e:
        print(f"Error during retraining: {e}")
        import traceback
        traceback.print_exc()
        return model

def plot_history(history, output_path):
    """Plot and save training history"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
    
    # Plot loss
    ax1.plot(history['train_loss'], 'b-', label='Training Loss')
    ax1.plot(history['val_loss'], 'r-', label='Validation Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot accuracy
    ax2.plot(history['train_acc'], 'b-', label='Training Accuracy')
    ax2.plot(history['val_acc'], 'r-', label='Validation Accuracy')
    ax2.set_title('Training and Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Training history plot saved to {output_path}")

def load_model(model_path):
    """
    Load a PyTorch model from file.
    
    Args:
        model_path: Path to the PyTorch model file
        
    Returns:
        PyTorch model
    """
    try:
        device = torch.device("cpu")
        
        # Try loading as a full model first
        try:
            model = torch.load(model_path, map_location=device)
            print(f"Loaded model from {model_path}")
            return model
        except:
            # If that fails, try loading as a state dict
            print("Could not load as full model, trying as state dict...")
            state_dict = torch.load(model_path, map_location=device)
            
            # Try to determine model type and create model
            if any("resnet18" in k for k in state_dict.keys()):
                model = models.resnet18(weights=None)
                model.load_state_dict(state_dict)
                return model
            elif any("resnet50" in k for k in state_dict.keys()):
                model = models.resnet50(weights=None)
                model.load_state_dict(state_dict)
                return model
            else:
                print("Could not determine model architecture from state dict.")
                return None
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description='Retrain a pruned model to recover accuracy')
    parser.add_argument('--model', type=str, required=True,
                        help='Path to the pruned model file')
    parser.add_argument('--data-dir', type=str, required=True,
                        help='Path to training data directory')
    parser.add_argument('--output', type=str, default='retrained_model',
                        help='Output directory for the retrained model')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of epochs for retraining')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='Batch size for retraining')
    parser.add_argument('--learning-rate', type=float, default=0.001,
                        help='Learning rate for retraining')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='Momentum for SGD optimizer')
    parser.add_argument('--weight-decay', type=float, default=1e-4,
                        help='Weight decay for regularization')
    
    args = parser.parse_args()

    # Create output directory if it doesn't exist
    os.makedirs(args.output, exist_ok=True)

    # Load the pruned model
    pruned_model = load_model(args.model)
    
    if pruned_model is None:
        print("Error: Could not load pruned model. Exiting.")
        return

    # Load training data
    data_loaders = load_training_data(
        args.data_dir, 
        batch_size=args.batch_size
    )
    
    if data_loaders[0] is None:
        print("Error: Could not load training data. Exiting.")
        return
    
    # Retrain the model
    retrained_model = retrain_model(
        pruned_model,
        data_loaders,
        epochs=args.epochs,
        lr=args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        output_dir=args.output
    )
    
    print("\nRetraining completed!")

if __name__ == "__main__":
    main()