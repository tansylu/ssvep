import os
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib.pyplot as plt
from datasets import load_dataset
import tempfile
from pathlib import Path
import torch.nn.utils.prune as prune

# --- CONFIG ---
# Path to the text file containing filters to prune
# Each line in the file should be in the format: "layer_id,filter_id"
filters_file_path = "filters_to_prune.txt"

pth_path = "data/models/resnet18.pth"
dataset_name = "Prisma-Multimodal/segmented-imagenet1k-subset"  # Hugging Face dataset name
# Cache directory for datasets - this will store the dataset locally after first download
# so it won't be redownloaded on subsequent runs
dataset_cache_dir = os.path.join(tempfile.gettempdir(), "hf_datasets_cache")
batch_size = 64
num_epochs = 10
learning_rate = 5e-4  # Reduced learning rate for more stable training
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint_dir = "checkpoints"  # Directory to save intermediate models

# Print GPU information
if torch.cuda.is_available():
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    # Don't set default tensor type to CUDA as it can cause issues with DataLoader
    # Instead, we'll explicitly move tensors to the device when needed
else:
    print("No GPU available, using CPU")

# --- Helper functions ---
def read_filters_from_file(file_path):
    """
    Read filters to prune from a text file.

    Args:
        file_path: Path to the text file containing filters to prune

    Returns:
        List of (layer_id, filter_id) tuples

    Format:
        Each line in the file should be in the format: "layer_id,filter_id"
        Lines starting with # are treated as comments and ignored
        Empty lines are ignored
    """
    filters_to_prune = []

    try:
        with open(file_path, 'r') as f:
            for line in f:
                # Skip comments and empty lines
                line = line.strip()
                if not line or line.startswith('#'):
                    continue

                try:
                    # Parse the line as "layer_id,filter_id"
                    parts = line.split(',')
                    if len(parts) != 2:
                        print(f"Warning: Invalid line format in {file_path}: '{line}'. Expected 'layer_id,filter_id'")
                        continue

                    layer_id = int(parts[0].strip())
                    filter_id = int(parts[1].strip())
                    filters_to_prune.append((layer_id, filter_id))
                except ValueError as e:
                    print(f"Warning: Error parsing line in {file_path}: '{line}'. {str(e)}")
                    continue

        print(f"Read {len(filters_to_prune)} filters to prune from {file_path}")
    except FileNotFoundError:
        print(f"Warning: Filters file not found: {file_path}")
    except Exception as e:
        print(f"Error reading filters file {file_path}: {str(e)}")

    return filters_to_prune

# --- Helper functions for model layer mapping ---
def get_model_layer_mapping(model):
    """
    Create a mapping of layer IDs to actual model modules using the custom indexing scheme.
    Returns:
    - conv_layers: List of (name, module, layer_id) tuples
    - layer_id_to_module: Dictionary mapping layer_id to module
    - module_to_layer_id: Dictionary mapping module to layer_id
    """
    # Get all Conv2d layers with their names, using the same indexing scheme as in src/core/model.py
    # In this scheme, we skip downsample layers when registering hooks, but layer IDs still follow the full sequence
    conv_layers = []
    idx = 0

    # Define the indices to skip (downsample layers)
    skip_indices = [7, 12, 17]  # These are the downsample layers in your scheme

    # First, collect all Conv2d layers
    all_conv_layers = [(name, m) for name, m in model.named_modules() if isinstance(m, nn.Conv2d)]

    # Then assign indices according to your scheme
    for i, (name, m) in enumerate(all_conv_layers):
        if 'downsample' in name:
            # This is a downsample layer, mark it as skipped
            conv_layers.append((name, m, f"skip_{i}"))
        else:
            # This is a regular layer, assign the next available index that's not in skip_indices
            while idx in skip_indices:
                idx += 1
            conv_layers.append((name, m, idx))
            idx += 1

    # Print all available convolutional layers for debugging
    print("\nAvailable convolutional layers in the model (using custom indexing scheme):")
    print("Layer ID | Layer Name | Is Downsample | Filters")
    print("-" * 70)
    for name, m, idx in conv_layers:
        is_downsample = not isinstance(idx, int)
        filter_count = m.out_channels
        if isinstance(idx, int):
            print(f"Layer {idx:2d} | {name:30s} | {'Yes' if is_downsample else 'No ':11s} | {filter_count:4d}")
        else:
            print(f"Skipped  | {name:30s} | {'Yes' if is_downsample else 'No ':11s} | {filter_count:4d}")

    # Create dictionaries for easy lookup
    layer_id_to_module = {}
    module_to_layer_id = {}

    for name, module, idx in conv_layers:
        if isinstance(idx, int):  # Only include non-skipped layers
            layer_id_to_module[idx] = module
            module_to_layer_id[module] = idx

    return conv_layers, layer_id_to_module, module_to_layer_id

# --- Define HFDataset class ---
class HFDataset(Dataset):
    def __init__(self, hf_dataset, transform=None):
        self.dataset = hf_dataset
        self.transform = transform
        # Get class names from dataset features if available
        if hasattr(self.dataset, 'features') and 'label' in self.dataset.features:
            self.classes = self.dataset.features['label'].names
        else:
            # Otherwise try to get unique labels from available label fields
            label_field = None
            for field in ['label', 'class', 'imagenet_label']:
                if field in self.dataset.column_names:
                    label_field = field
                    break

            if label_field:
                # Sort the classes to ensure consistent mapping between runs
                self.classes = sorted(list(set(self.dataset[label_field])))
            else:
                print("Warning: No label field found in dataset. Using default 1000 classes.")
                self.classes = list(range(1000))

    def get_class_to_idx(self, class_name):
        """Helper method to get the index of a class by name"""
        if isinstance(class_name, str):
            # Try direct lookup first
            try:
                # Convert all classes to strings for comparison if needed
                class_list = [str(c) for c in self.classes]
                return class_list.index(class_name)
            except ValueError:
                # If that fails, try string comparison
                for i, c in enumerate(self.classes):
                    if str(c) == class_name:
                        return i
                # If we get here, the class wasn't found
                raise ValueError(f"Class '{class_name}' not found in class list")

        # If it's not a string but a number, make sure it's within valid range
        if isinstance(class_name, (int, float)):
            class_idx = int(class_name)
            if class_idx < 0 or class_idx >= len(self.classes):
                print(f"Warning: Label {class_idx} is out of range [0, {len(self.classes)-1}]")
                # Use a default label (0) to avoid crashing
                return 0
            return class_idx

        return class_name  # Return as is for other types

    def get_classes(self):
        """Return the list of classes"""
        return self.classes

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]

        # --- Get image ---
        if 'image' in item:
            image = item['image']
        elif 'img' in item:
            image = item['img']
        else:
            raise ValueError("Dataset doesn't contain 'image' or 'img' field")

        # --- Get label (handle different naming conventions) ---
        if 'label' in item:
            label = item['label']
        elif 'class' in item:
            label = item['class']
        elif 'imagenet_label' in item:
            label = item['imagenet_label']
        else:
            raise ValueError("Dataset doesn't contain a recognized label field. Available fields: " + str(list(item.keys())))

        if isinstance(label, str):
            try:
                # Use the helper method to get the class index
                label = self.get_class_to_idx(label)
            except ValueError as e:
                print(f"Warning: Error converting label '{label}': {str(e)}")
                # Use a default label (0) to avoid crashing
                label = 0

        # --- Convert label to scalar tensor ---
        if isinstance(label, torch.Tensor):
            if label.ndim > 0:
                label = label[0].long()
        elif isinstance(label, (list, tuple)):
            label = torch.tensor(label[0], dtype=torch.long) if len(label) > 0 else torch.tensor(0, dtype=torch.long)
        else:
            label = torch.tensor(label, dtype=torch.long)

        # Final validation to ensure label is within valid range
        if label.item() < 0 or label.item() >= len(self.classes):
            print(f"Warning: Final label {label.item()} is out of range [0, {len(self.classes)-1}]")
            # Use a default label (0) to avoid crashing
            label = torch.tensor(0, dtype=torch.long)

        # --- Apply image transform ---
        if self.transform:
            image = self.transform(image)

        return image, label

# --- Load dataset using Hugging Face datasets ---
def load_dataset_from_huggingface(dataset_name_or_path, split="train", transform=None, cache_dir=None):
    """Load a dataset from Hugging Face datasets.

    This function uses the Hugging Face datasets library to load a dataset.
    When cache_dir is provided, the dataset will be cached locally after the first download,
    so it won't be redownloaded on subsequent runs, significantly improving performance.
    """
    # Map split names to HF dataset splits
    split_mapping = {
        "train": "train",
        "val": "validation"
    }

    # Load the dataset
    hf_split = split_mapping.get(split, split)
    try:
        print(f"Loading dataset from Hugging Face Hub: {dataset_name_or_path}, split: {hf_split}")
        # Use cache_dir if provided to avoid redownloading
        dataset = load_dataset(dataset_name_or_path, split=hf_split, cache_dir=cache_dir)
        print(f"Successfully loaded dataset with {len(dataset)} examples")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        raise e

    return HFDataset(dataset, transform)

# --- Pruning and Fine-tuning ---
def main():
    """Main function for pruning and fine-tuning.

    This implementation optimizes dataset loading by:
    1. Using a persistent cache directory to store datasets after first download
    2. Loading each dataset split only once and reusing it
    3. Applying transforms to the cached datasets

    This avoids redownloading the dataset on every run, significantly improving performance.
    """
    # Make sure we're using the GPU if available
    if torch.cuda.is_available():
        # Clear CUDA cache at the beginning to start with clean memory
        torch.cuda.empty_cache()
        print("Starting with clean GPU memory")

    # Create cache directory if it doesn't exist
    os.makedirs(dataset_cache_dir, exist_ok=True)
    print(f"Using dataset cache directory: {dataset_cache_dir}")

    # --- Define transforms for training and evaluation ---
    # Define transforms for training (with augmentation)
    train_transform = transforms.Compose([
        transforms.Lambda(lambda img: img.convert("RGB")),  # Ensure RGB
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    eval_transform = transforms.Compose([
        transforms.Lambda(lambda img: img.convert("RGB")),  # Ensure RGB
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    # --- Load datasets once and reuse ---
    print("Loading datasets from Hugging Face (will be cached for future runs)...")

    # Load datasets once and reuse them
    try:
        # Use the validation set as training set (it has 50,000 examples)
        # Load without transform first to determine number of classes
        validation_dataset = load_dataset_from_huggingface(
            dataset_name,
            split="validation",
            cache_dir=dataset_cache_dir
        )
        num_classes = len(validation_dataset.classes)
        print(f"Found {num_classes} classes in the dataset")

        # Create training dataset with transform
        train_dataset = HFDataset(validation_dataset.dataset, transform=train_transform)
        train_dataset.classes = validation_dataset.classes  # Copy class information
        print(f"Prepared {len(train_dataset)} training images (from validation split)")

        # Load training set as validation set (it has 10,000 examples)
        train_split_dataset = load_dataset_from_huggingface(
            dataset_name,
            split="train",
            cache_dir=dataset_cache_dir
        )

        # Create validation dataset with transform
        val_dataset = HFDataset(train_split_dataset.dataset, transform=eval_transform)
        val_dataset.classes = train_split_dataset.classes  # Copy class information
        print(f"Prepared {len(val_dataset)} validation images (from train split)")
    except Exception as e:
        print(f"Warning: Could not load datasets: {e}")
        print("Using default 1000 classes for ImageNet.")
        num_classes = 1000
        return  # Exit if we can't load the datasets

    # --- Load pretrained model ---
    model = torchvision.models.resnet18(weights=None)

    # First move the model to the target device
    model = model.to(device)

    # Modify the final fully connected layer to match the number of classes in your dataset
    model.fc = nn.Linear(model.fc.in_features, num_classes).to(device)

    # Load pretrained weights directly to the target device
    try:
        # First try loading with map_location=device
        state_dict = torch.load(pth_path, map_location=device)
        model.load_state_dict(state_dict, strict=False)  # strict=False to ignore fc layer mismatch
    except RuntimeError as e:
        print(f"Error loading model with map_location=device: {e}")
        print("Trying alternative loading method...")

        # If that fails, try loading to CPU first, then manually move each tensor to the device
        state_dict = torch.load(pth_path, map_location='cpu')

        # Manually move each tensor to the correct device
        for key in state_dict:
            state_dict[key] = state_dict[key].to(device)

        model.load_state_dict(state_dict, strict=False)

    print(f"Model loaded and moved to {device}")

    # --- Prune filters ---
    print("Pruning filters...")

    # Read filters to prune from the text file
    filters_to_prune = read_filters_from_file(filters_file_path)
    print(f"\nUsing direct pruning with {len(filters_to_prune)} filters read from {filters_file_path}...")

    # Function to apply unstructured pruning to specific filters
    def apply_direct_pruning(model, filters_to_prune_list):
        """
        Apply pruning to specific filters in specific layers using PyTorch's pruning utilities.
        
        Args:
            model: The PyTorch model to prune
            filters_to_prune_list: List of (layer_id, filter_id) tuples to prune
        """
        if not filters_to_prune_list:
            print("No filters specified for pruning. Skipping pruning step.")
            return

        # Get layer mapping
        _, layer_id_to_module, _ = get_model_layer_mapping(model)

        # Group filters by layer for more efficient pruning
        filters_by_layer = {}
        for layer_id, filter_id in filters_to_prune_list:
            if layer_id not in filters_by_layer:
                filters_by_layer[layer_id] = []
            filters_by_layer[layer_id].append(filter_id)

        # Count total filters to prune
        total_to_prune = len(filters_to_prune_list)
        pruned_count = 0

        print(f"\nPruning {total_to_prune} filters as specified in the filters_to_prune list")

        # Apply pruning to each layer
        for layer_id, filter_ids in filters_by_layer.items():
            if layer_id in layer_id_to_module:
                module = layer_id_to_module[layer_id]

                # Check if filter IDs are valid
                valid_filter_ids = [f_id for f_id in filter_ids if f_id < module.out_channels]
                invalid_count = len(filter_ids) - len(valid_filter_ids)

                if invalid_count > 0:
                    print(f"WARNING: {invalid_count} invalid filter IDs for layer {layer_id} (max valid ID: {module.out_channels-1})")

                if valid_filter_ids:
                    # Format the filter IDs list for better readability
                    if len(valid_filter_ids) > 20:
                        filter_ids_str = str(valid_filter_ids[:10])[:-1] + ", ... " + str(valid_filter_ids[-10:])[1:]
                    else:
                        filter_ids_str = str(valid_filter_ids)

                    print(f"Pruning Layer {layer_id}: {len(valid_filter_ids)} filters with indices: {filter_ids_str}")

                    # Create a mask where pruned filters are 0 and kept filters are 1
                    mask = torch.ones_like(module.weight)
                    for filter_id in valid_filter_ids:
                        mask[filter_id] = 0
                    
                    # Apply custom pruning with the created mask
                    prune.custom_from_mask(module, name='weight', mask=mask)
                    
                    # If there's a bias, prune it too
                    if module.bias is not None and hasattr(module.bias, 'size'):
                        bias_mask = torch.ones_like(module.bias)
                        for filter_id in valid_filter_ids:
                            if filter_id < bias_mask.size(0):
                                bias_mask[filter_id] = 0
                        prune.custom_from_mask(module, name='bias', mask=bias_mask)
                    
                    pruned_count += len(valid_filter_ids)
                    print(f"Pruned {len(valid_filter_ids)} filters from Layer {layer_id} (kept {module.out_channels - len(valid_filter_ids)})")
            else:
                print(f"WARNING: Layer ID {layer_id} not found in model. Skipping {len(filter_ids)} filters.")

        print(f"\nPruning summary: Applied masks to {pruned_count} filters as specified")
        print("These filters will remain pruned during fine-tuning")

    # Apply direct pruning using the list provided at the top of the file
    apply_direct_pruning(model, filters_to_prune)

    # --- Dataloaders ---
    # Datasets have already been loaded and prepared with transforms

    # Ensure model's output layer matches the number of classes in the dataset
    if model.fc.out_features != len(train_dataset.classes):
        print(f"Adjusting model's FC layer from {model.fc.out_features} to {len(train_dataset.classes)} classes")
        model.fc = nn.Linear(model.fc.in_features, len(train_dataset.classes))
        model.to(device)
        num_classes = len(train_dataset.classes)
    else:
        num_classes = model.fc.out_features

    # Create DataLoader with multiple workers and pin_memory for faster GPU transfer
    num_workers = min(4, os.cpu_count() or 1)  # Use up to 4 workers or available CPU cores

    # Create a generator for shuffling
    # DataLoader with num_workers > 0 requires a CPU generator
    g = torch.Generator()

    # Seed the generator for reproducibility
    g.manual_seed(42)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False,
        generator=g
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False,
        generator=g  # Use the same generator for consistency
    )

    # --- Fine-tuning ---
    criterion = nn.CrossEntropyLoss().to(device)  # Move loss function to GPU
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

    train_acc_list, val_acc_list = [], []

    # Create checkpoint directory if it doesn't exist
    os.makedirs(checkpoint_dir, exist_ok=True)
    best_val_acc = 0.0

    for epoch in range(num_epochs):
        model.train()
        correct, total, running_loss = 0, 0, 0.0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)

            # Validate targets are within range
            if targets.max() >= num_classes:
                print(f"Warning: Found labels outside valid range: max={targets.max().item()}, valid range=[0,{num_classes-1}]")
                # Filter out invalid targets
                valid_mask = targets < num_classes
                if valid_mask.sum() == 0:
                    print("No valid targets in this batch, skipping")
                    continue
                inputs = inputs[valid_mask]
                targets = targets[valid_mask]

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
                inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)

                # Validate targets are within range
                if targets.max() >= num_classes:
                    print(f"Warning: Found labels outside valid range in validation: max={targets.max().item()}, valid range=[0,{num_classes-1}]")
                    # Filter out invalid targets
                    valid_mask = targets < num_classes
                    if valid_mask.sum() == 0:
                        print("No valid targets in this validation batch, skipping")
                        continue
                    inputs = inputs[valid_mask]
                    targets = targets[valid_mask]

                outputs = model(inputs)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        val_acc = 100. * correct / total
        val_acc_list.append(val_acc)

        print(f"Epoch [{epoch+1}/{num_epochs}] Loss: {running_loss:.4f} | Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%")

        # Save checkpoint after each epoch
        checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch+1}.pth")

        # Clear CUDA cache before saving to free up memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_acc': train_acc,
            'val_acc': val_acc,
            'train_acc_history': train_acc_list,
            'val_acc_history': val_acc_list,
            'device': str(device)  # Save device information
        }, checkpoint_path)
        print(f"Checkpoint saved to {checkpoint_path}")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_path = os.path.join(checkpoint_dir, "best_model.pth")
            # Save best model to CPU for compatibility
            model_cpu = model.to('cpu')
            torch.save(model_cpu, best_model_path)
            # Move model back to original device
            model = model.to(device)
            print(f"New best model saved with validation accuracy: {val_acc:.2f}%")

    # --- Save final model ---
    model.zero_grad()  # Clear gradients to reduce file size
    model_filename = "resnet18_segmented_imagenet_finetuned.pth"

    # Move model to CPU before saving to ensure compatibility
    model_cpu = model.to('cpu')
    torch.save(model_cpu, model_filename)
    print(f"Model saved to {model_filename}")

    # Move model back to original device
    model = model.to(device)

    # Also save final checkpoint with all training history
    final_checkpoint = os.path.join(checkpoint_dir, "final_checkpoint.pth")
    torch.save({
        'epochs_completed': num_epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'final_train_acc': train_acc_list[-1],
        'final_val_acc': val_acc_list[-1],
        'train_acc_history': train_acc_list,
        'val_acc_history': val_acc_list,
        'best_val_acc': best_val_acc,
        'device': str(device)  # Save device information
    }, final_checkpoint)
    print(f"Final checkpoint saved to {final_checkpoint}")

    # Clear CUDA cache to free up memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # --- Plot accuracy ---
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
    import multiprocessing
    multiprocessing.freeze_support()

    # Set PyTorch to use deterministic algorithms for reproducibility
    # Comment these out if they cause performance issues
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Run with higher precision if available
    if torch.cuda.is_available():
        # Enable TF32 precision on Ampere GPUs (A100, RTX 30xx, etc.)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        print("TF32 precision enabled for compatible GPUs")

        # Set CUDA device to device 0
        torch.cuda.set_device(0)

    main()
