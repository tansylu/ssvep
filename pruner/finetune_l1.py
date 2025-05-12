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

# --- CONFIG ---
pth_path = "data/models/resnet18_weights_only.pth"
dataset_name = "Prisma-Multimodal/segmented-imagenet1k-subset"  # Hugging Face dataset name
# Cache directory for datasets - this will store the dataset locally after first download
# so it won't be redownloaded on subsequent runs
dataset_cache_dir = os.path.join(tempfile.gettempdir(), "hf_datasets_cache")
batch_size = 64
num_epochs = 10
learning_rate = 1e-3  # Increased from 5e-4 to 1e-3
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint_dir = "checkpoints"  # Directory to save intermediate models
pruning_percentage = 0.1  # Default to 10% pruning

# Print GPU information
if torch.cuda.is_available():
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    # Don't set default tensor type to CUDA as it can cause issues with DataLoader
    # Instead, we'll explicitly move tensors to the device when needed
else:
    print("No GPU available, using CPU")

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

def evaluate_model(model, data_loader, device):
    """
    Evaluate model accuracy on a dataset.

    Args:
        model: PyTorch model to evaluate
        data_loader: DataLoader with evaluation data
        device: Device to run evaluation on

    Returns:
        float: Accuracy percentage
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
    return accuracy

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
        dataset = load_dataset(dataset_name_or_path, split=hf_split, cache_dir=cache_dir, trust_remote_code=True)
        print(f"Successfully loaded dataset with {len(dataset)} examples")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        raise e

    return HFDataset(dataset, transform)

# Add this function to map dataset classes to ImageNet classes
def map_dataset_to_imagenet_classes(dataset):
    """
    Map dataset classes to match ImageNet class indices using improved string matching.

    Args:
        dataset: HFDataset with classes attribute

    Returns:
        Dictionary mapping dataset indices to ImageNet indices
    """
    # Load ImageNet class mappings
    try:
        # Try to get ImageNet class names from torchvision
        imagenet_classes = torchvision.models.ResNet18_Weights.IMAGENET1K_V1.meta['categories']
    except:
        print("Could not load ImageNet class names from torchvision, using indices only")
        return {i: i for i in range(len(dataset.classes))}

    # Create mapping dictionary
    class_mapping = {}

    # For each class in the dataset, find the best matching ImageNet class
    for i, class_name in enumerate(dataset.classes):
        class_mapping[i] = i  # Default to same index

        # Skip if class_name is not a string
        if not isinstance(class_name, str):
            continue

        # Normalize class name for better matching
        normalized_name = class_name.lower().replace('_', ' ').strip()

        # Try exact match first
        for j, imagenet_class in enumerate(imagenet_classes):
            if normalized_name == imagenet_class.lower():
                class_mapping[i] = j
                print(f"Exact match: {class_name} -> {imagenet_class} (index {j})")
                break

        # If no exact match, try substring match
        if class_mapping[i] == i:
            best_match = None
            best_score = 0

            for j, imagenet_class in enumerate(imagenet_classes):
                # Check if dataset class is a substring of ImageNet class or vice versa
                if normalized_name in imagenet_class.lower() or imagenet_class.lower() in normalized_name:
                    # Calculate match score based on length of common substring
                    score = len(set(normalized_name.split()) & set(imagenet_class.lower().split()))
                    if score > best_score:
                        best_score = score
                        best_match = (j, imagenet_class)

            if best_match and best_score > 0:
                class_mapping[i] = best_match[0]
                print(f"Substring match: {class_name} -> {best_match[1]} (index {best_match[0]})")

    print(f"Created mapping for {len(class_mapping)} classes")
    return class_mapping

# Add this function to print directory contents
def print_directory_contents(dataset):
    """
    Print information about the dataset structure to help debug class mapping issues.

    Args:
        dataset: HFDataset instance
    """
    print("\n=== Dataset Structure Information ===")

    # Print dataset features if available
    if hasattr(dataset.dataset, 'features'):
        print("Dataset features:")
        for feature_name, feature in dataset.dataset.features.items():
            print(f"  {feature_name}: {type(feature).__name__}")

    # Print sample data point
    print("\nSample data point:")
    sample = dataset.dataset[0]
    for key, value in sample.items():
        value_type = type(value).__name__
        value_shape = getattr(value, 'shape', None)
        value_preview = str(value)[:100] + '...' if len(str(value)) > 100 else value
        print(f"  {key} ({value_type}): {value_shape if value_shape else ''} {value_preview}")

    # Try to get class information
    print("\nClass information:")
    print(f"  Number of classes: {len(dataset.classes)}")
    print(f"  First 10 classes: {dataset.classes[:10]}")

    # If the dataset has a 'label' field, check its distribution
    if 'label' in dataset.dataset.column_names:
        labels = [item['label'] for item in dataset.dataset]
        unique_labels = set(labels)
        print(f"  Unique label values: {len(unique_labels)}")
        print(f"  Label range: {min(unique_labels)} to {max(unique_labels)}")

    # If the dataset has an 'image' field, check image properties
    if 'image' in dataset.dataset.column_names:
        # Get the first image
        sample_image = dataset.dataset[0]['image']
        print("\nImage information:")
        print(f"  Image type: {type(sample_image).__name__}")
        if hasattr(sample_image, 'size'):
            print(f"  Image size: {sample_image.size}")
        if hasattr(sample_image, 'mode'):
            print(f"  Image mode: {sample_image.mode}")

    print("=" * 40)

# Create a custom dataset class that applies the mapping
class MappedDataset(Dataset):
    """
    Dataset wrapper that applies a class mapping to the target labels.

    Args:
        dataset: Original dataset
        class_mapping: Dictionary mapping original class indices to new indices
    """
    def __init__(self, dataset, class_mapping):
        self.dataset = dataset
        self.class_mapping = class_mapping
        self.classes = dataset.classes  # Keep original classes for reference

    def __getitem__(self, index):
        data, target = self.dataset[index]

        # Apply class mapping if the target is within range
        if isinstance(target, (int, torch.Tensor)) and target.item() if isinstance(target, torch.Tensor) else target < len(self.class_mapping):
            target_idx = target.item() if isinstance(target, torch.Tensor) else target
            mapped_target = self.class_mapping.get(target_idx, target_idx)

            # Convert back to tensor if it was a tensor
            if isinstance(target, torch.Tensor):
                mapped_target = torch.tensor(mapped_target, dtype=target.dtype, device=target.device)

            return data, mapped_target

        return data, target

    def __len__(self):
        return len(self.dataset)

# --- L1 Pruning Implementation ---
def apply_l1_pruning(model, pruning_percentage):
    """
    Apply L1 pruning to all convolutional layers in the model.

    This implementation calculates L1 norm for each filter and prunes
    the filters with the lowest L1 norms directly by zeroing them out.

    Args:
        model: The PyTorch model to prune
        pruning_percentage: Percentage of filters to prune (0-1)

    Returns:
        Dictionary with pruning statistics
    """
    if pruning_percentage <= 0 or pruning_percentage >= 1:
        print(f"Warning: Invalid pruning percentage {pruning_percentage}. Must be between 0 and 1.")
        return {}

    # Get layer mapping
    conv_layers, _, _ = get_model_layer_mapping(model)

    # Apply L1 pruning to each convolutional layer
    pruned_layers = []
    for name, module, idx in conv_layers:
        if isinstance(idx, int):  # Only prune non-skipped layers
            # Calculate L1 norm for each filter
            with torch.no_grad():
                # Calculate L1 norm for each filter (sum of absolute values)
                filter_norms = torch.sum(torch.abs(module.weight.data), dim=(1, 2, 3))

                # Determine number of filters to prune
                num_filters = module.out_channels
                num_to_prune = int(num_filters * pruning_percentage)

                if num_to_prune > 0:
                    # Get indices of filters with lowest L1 norms
                    _, indices = torch.topk(filter_norms, k=num_filters - num_to_prune, largest=True)
                    mask = torch.ones(num_filters, device=module.weight.device)
                    mask[indices] = 0
                    prune_indices = torch.nonzero(mask).squeeze()

                    # Zero out the weights of pruned filters
                    for filter_idx in prune_indices:
                        module.weight.data[filter_idx] = 0
                        if module.bias is not None:
                            module.bias.data[filter_idx] = 0

                    # Instead of using hooks (which can cause pickling issues),
                    # we'll create a pruning mask that will be applied during training

                    # Store the pruned indices as a module attribute
                    module._pruned_indices = prune_indices.cpu().tolist() if hasattr(prune_indices, 'cpu') else prune_indices

                    # Create a forward pre-hook to zero out the weights before each forward pass
                    def zero_pruned_filters(module, _):
                        with torch.no_grad():
                            for filter_idx in module._pruned_indices:
                                module.weight.data[filter_idx].fill_(0)
                                if module.bias is not None:
                                    module.bias.data[filter_idx].fill_(0)
                        return None

                    # Register the pre-hook
                    module.register_forward_pre_hook(zero_pruned_filters)

                    pruned_layers.append((name, idx, num_filters, num_to_prune))

    # Count total parameters after pruning
    zero_params = sum((p == 0).sum().item() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    actual_pruning_percentage = zero_params / total_params * 100

    # Print pruning summary
    print("\nPruning Summary:")
    print(f"Applied L1 pruning with target percentage: {pruning_percentage*100:.1f}%")
    print(f"Actual pruning percentage achieved: {actual_pruning_percentage:.2f}%")
    print(f"Total parameters: {total_params:,}")
    print(f"Zero-valued parameters: {zero_params:,}")

    print("\nPruned layers:")
    print("Layer Name | Layer ID | Total Filters | Pruned Filters")
    print("-" * 70)
    for name, idx, total_filters, pruned_filters in pruned_layers:
        print(f"{name:30s} | {idx:8d} | {total_filters:13d} | {pruned_filters:14d}")

    return {
        "target_percentage": pruning_percentage * 100,
        "actual_percentage": actual_pruning_percentage,
        "total_params": total_params,
        "zero_params": zero_params,
        "pruned_layers": pruned_layers
    }

def ensure_rgb(img):
    return img.convert("RGB")

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

    # Define transforms for training (with augmentation)
    train_transform = transforms.Compose([
        transforms.Lambda(ensure_rgb),  # Ensure RGB
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    eval_transform = transforms.Compose([
        transforms.Lambda(ensure_rgb),  # Ensure RGB
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

    # Print dataset structure information
    print_directory_contents(validation_dataset)

    # Map dataset classes to ImageNet classes using improved string matching
    class_mapping = map_dataset_to_imagenet_classes(validation_dataset)


    # Wrap datasets with mapping if available
    if class_mapping:
        train_dataset = MappedDataset(train_dataset, class_mapping)
        val_dataset = MappedDataset(val_dataset, class_mapping)
        print("Applied class mapping to datasets")

    # --- Load pretrained model ---
    model = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)

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

    # Dataloaders have already been created with the correct transforms

    # Create DataLoader with multiple workers and pin_memory for faster GPU transfer
    num_workers = 0  # Use up to 4 workers or available CPU cores

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
        pin_memory=False,
        generator=g
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
        generator=g  # Use the same generator for consistency
    )

    # Evaluate model accuracy before pruning
    print("\nEvaluating model accuracy before pruning...")
    pre_pruning_accuracy = evaluate_model(model, val_loader, device)
    print(f"Pre-pruning accuracy: {pre_pruning_accuracy:.2f}%")

    # --- Apply L1 pruning ---
    print("\nApplying L1 pruning...")
    pruning_stats = apply_l1_pruning(model, pruning_percentage)

    # Evaluate model accuracy after pruning but before fine-tuning
    print("\nEvaluating model accuracy after pruning (before fine-tuning)...")
    post_pruning_accuracy = evaluate_model(model, val_loader, device)
    print(f"Post-pruning accuracy: {post_pruning_accuracy:.2f}%")
    print(f"Accuracy drop: {pre_pruning_accuracy - post_pruning_accuracy:.2f}%")

    # --- Fine-tuning ---
    criterion = nn.CrossEntropyLoss().to(device)  # Move loss function to GPU
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=learning_rate,
        momentum=0.9,
        weight_decay=5e-4  # Added weight decay for better generalization
    )

    # Add a learning rate scheduler to gradually reduce the learning rate
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=3,  # Reduce learning rate every 3 epochs
        gamma=0.1     # Reduce to 10% of previous value
    )

    train_acc_list, val_acc_list = [], []
    best_val_acc = 0.0

    for epoch in range(num_epochs):
        model.train()
        print(f"Starting epoch {epoch+1}/{num_epochs}")
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
            'device': str(device),  # Save device information
            'pruning_stats': pruning_stats  # Save pruning statistics
        }, checkpoint_path)
        print(f"Checkpoint saved to {checkpoint_path}")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_path = os.path.join(checkpoint_dir, "best_model_l1.pth")
            # Save best model to CPU for compatibility
            model_cpu = model.to('cpu')
            torch.save(model_cpu, best_model_path)
            # Move model back to original device
            model = model.to(device)
            print(f"New best model saved with validation accuracy: {val_acc:.2f}%")

        # Step the scheduler
        scheduler.step()

    # --- Save final model ---
    model.zero_grad()  # Clear gradients to reduce file size
    percentage_str = f"{int(pruning_percentage * 100)}"
    model_filename = os.path.join(checkpoint_dir, f"resnet18_l1_pruned_{percentage_str}pct.pth")

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
        'device': str(device),  # Save device information
        'pruning_stats': pruning_stats,  # Save pruning statistics
        'pre_pruning_accuracy': pre_pruning_accuracy,
        'post_pruning_accuracy': post_pruning_accuracy
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
    plt.title(f"Accuracy over Fine-Tuning Epochs (L1 {percentage_str}% Pruning)")
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(checkpoint_dir, f"accuracy_plot_{percentage_str}pct.png"))
    plt.close()

    print(f"\nTraining completed for L1 {percentage_str}% pruning.")
    print(f"Pre-pruning accuracy: {pre_pruning_accuracy:.2f}%")
    print(f"Post-pruning accuracy (before fine-tuning): {post_pruning_accuracy:.2f}%")
    print(f"Final accuracy (after fine-tuning): {val_acc_list[-1]:.2f}%")
    print(f"Best validation accuracy: {best_val_acc:.2f}%")
    print(f"All outputs saved to: {checkpoint_dir}")

if __name__ == "__main__":
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
