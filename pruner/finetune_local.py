import os
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision.datasets import ImageFolder
import matplotlib.pyplot as plt
import torch.nn.utils.prune as prune

# --- CONFIG ---
local_dataset_path = "data\\10k-imagenet\\imagenet_subtrain" # Path to your local dataset
class_mapping_file = "data\\LOC_synset_mapping.txt" 
batch_size = 64
num_epochs = 10
learning_rate = 1e-3
train_split_ratio = 0.8  # Percentage of data to use for training
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint_dir = "checkpoints"  # Directory to save intermediate models

def load_class_mapping(mapping_file):
    """
    Load class mapping from a file.
    Args:
        mapping_file (str): Path to the mapping file.
    Returns:
        dict: Mapping from WordNet IDs to class names or indices.
    """
    mapping = {}
    with open(mapping_file, "r") as f:
        for line in f:
            # Skip empty lines or lines that don't contain a comma
            if not line.strip() or "," not in line:
                continue
            
            # Split the line into WordNet ID and class names
            parts = line.strip().split(" ", 1)  # Split only at the first comma
            wordnet_id = parts[0]
            class_name = parts[1]  # Join the rest as the class name
            mapping[wordnet_id] = class_name
    return mapping

class MappedImageFolder(ImageFolder):
    """
    Custom ImageFolder that maps WordNet IDs to human-readable class names.
    """
    def __init__(self, root, transform, class_mapping):
        super().__init__(root, transform)
        self.class_mapping = class_mapping
        self.classes = [self.class_mapping.get(c, c) for c in self.classes]
        self.class_to_idx = {self.class_mapping.get(k, k): v for k, v in self.class_to_idx.items()}


# Print GPU information
if torch.cuda.is_available():
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
else:
    print("No GPU available, using CPU")

# --- Helper Functions ---
def evaluate_model(model, data_loader, device):
    """
    Evaluate model accuracy on a dataset.
    """
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    accuracy = 100. * correct / total
    
    # Debugging: Print predicted and target values for a batch in the validation set
    model.eval()
    return accuracy

def main():
    """Main function for pruning and fine-tuning."""
    # Define transforms for training and evaluation
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    eval_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Load class mapping
    print("Loading class mapping...")
    class_mapping = load_class_mapping(class_mapping_file)

    # Load the dataset using MappedImageFolder
    print("Loading local dataset...")
    full_dataset = MappedImageFolder(root=local_dataset_path, transform=train_transform, class_mapping=class_mapping)
    print("Class Mapping:")
    for wordnet_id, class_name in list(class_mapping.items())[:10]:  # Print first 10 mappings
        print(f"{wordnet_id}: {class_name}")
    # Split the dataset into training and validation sets
    train_size = int(len(full_dataset) * train_split_ratio)
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    # Apply evaluation transforms to the validation dataset
    val_dataset.dataset.transform = eval_transform

    print(f"Found {len(full_dataset.classes)} classes in the dataset.")
    print(f"Split dataset into {len(train_dataset)} training images and {len(val_dataset)} validation images.")

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # Load pretrained model
    model = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
    
    model = model.to(device)

    # Modify the final fully connected layer to match the number of classes
    num_classes = len(full_dataset.classes)
    model.fc = nn.Linear(model.fc.in_features, num_classes).to(device)

    # Load additional weights if available
    pth_path = "data/models/resnet18_weights_only.pth"  # Path to the .pth file
    try:
        # Load weights with map_location=device
        state_dict = torch.load(pth_path, map_location=device)
        model.load_state_dict(state_dict, strict=False)  # strict=False to ignore fc layer mismatch
        print("Additional weights loaded successfully.")
    except RuntimeError as e:
        print(f"Error loading weights with map_location=device: {e}")
        print("Trying alternative loading method...")

        # If loading fails, load to CPU first and manually move tensors to the device
        state_dict = torch.load(pth_path, map_location='cpu')
        for key in state_dict:
            state_dict[key] = state_dict[key].to(device)
        model.load_state_dict(state_dict, strict=False)
        print("Weights loaded using alternative method.")

    print(f"Model loaded and moved to {device}")

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    # Evaluate model accuracy before pruning
    print("\nEvaluating model accuracy before pruning...")
   
    pre_pruning_accuracy = evaluate_model(model, val_loader, device)
    print(f"Pre-pruning accuracy: {pre_pruning_accuracy:.2f}%")

    # --- Fine-tuning ---
    train_acc_list, val_acc_list = [], []
    os.makedirs(checkpoint_dir, exist_ok=True)
    best_val_acc = 0.0
    import matplotlib.pyplot as plt
    for epoch in range(num_epochs):
        model.train()
        correct, total, running_loss = 0, 0, 0.0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        train_acc = 100. * correct / total
        train_acc_list.append(train_acc)

        # Validation
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        val_acc = 100. * correct / total
        val_acc_list.append(val_acc)

        print(f"Epoch [{epoch+1}/{num_epochs}] Loss: {running_loss:.4f} | Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_path = os.path.join(checkpoint_dir, "best_model.pth")
            torch.save(model.state_dict(), best_model_path)
            print(f"New best model saved with validation accuracy: {val_acc:.2f}%")

        scheduler.step()

    # Save final model
    model_filename = "resnet18_finetuned.pth"
    torch.save(model.state_dict(), model_filename)
    print(f"Model saved to {model_filename}")

    # Plot accuracy
    plt.figure(figsize=(10, 6))
    plt.plot(train_acc_list, label="Train Accuracy")
    plt.plot(val_acc_list, label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title("Accuracy over Fine-Tuning Epochs")
    plt.legend()
    plt.grid()
    plt.savefig("accuracy_plot.png")
    plt.show()

if __name__ == "__main__":
    main()