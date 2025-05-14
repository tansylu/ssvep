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
import argparse

# --- CONFIG ---
# Default configuration values
DEFAULT_FILTERS_FILE_PATH = "data/filters_to_prune/filters_to_prune_10_percent.txt"
pth_path = "data/models/resnet18_weights_only.pth"
dataset_name = "Prisma-Multimodal/segmented-imagenet1k-subset"  # Hugging Face dataset name
# Cache directory for datasets - this will store the dataset locally after first download
# so it won't be redownloaded on subsequent runs
dataset_cache_dir = os.path.join(tempfile.gettempdir(), "hf_datasets_cache")
batch_size = 64
num_epochs = 10
learning_rate = 1e-3  # Increased from 5e-4 to 1e-3
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint_dir = "checkpoints"  # Base directory to save intermediate models

# Function to parse command-line arguments
def parse_args():
    """Parse command-line arguments for the script."""
    parser = argparse.ArgumentParser(description='Fine-tune a pruned ResNet18 model.')
    parser.add_argument('--filter_file', '--filters', type=str, default=DEFAULT_FILTERS_FILE_PATH,
                        help='Path to the text file containing filters to prune. Each line should be in the format: "layer_id,filter_id"')
    parser.add_argument('--epochs', type=int, default=num_epochs,
                        help=f'Number of training epochs (default: {num_epochs})')
    parser.add_argument('--batch-size', type=int, default=batch_size,
                        help=f'Batch size for training (default: {batch_size})')
    parser.add_argument('--learning-rate', '--lr', type=float, default=learning_rate,
                        help=f'Learning rate (default: {learning_rate})')
    parser.add_argument('--no-show-plot', action='store_true',
                        help='Do not display plots interactively (useful for batch processing)')
    return parser.parse_args()

# Function to get filter file basename (without path and extension)
def get_filter_file_basename(file_path):
    """Extract the base name of the filter file without path and extension."""
    return os.path.splitext(os.path.basename(file_path))[0]

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

    # Ensure model is on the correct device
    model = model.to(device)

    # Use larger batch size and non-blocking transfers for better GPU utilization
    with torch.no_grad():
        for inputs, targets in data_loader:
            # Use non_blocking=True for asynchronous transfer to GPU
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            # Forward pass
            outputs = model(inputs)
            _, predicted = outputs.max(1)

            # Update statistics
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    # Calculate accuracy
    accuracy = 100. * correct / total if total > 0 else 0
    return accuracy

# --- Define MappedDataset class ---
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
    dataset_to_imagenet = {}
    unmapped_classes = []

    print("\nMapping dataset classes to ImageNet classes...")

    # Preprocess ImageNet classes for better matching
    processed_imagenet_classes = []
    for idx, name in enumerate(imagenet_classes):
        # Convert to lowercase and remove punctuation
        processed = name.lower()
        processed = ''.join(c for c in processed if c.isalnum() or c.isspace())
        # Split into words for partial matching
        words = set(processed.split())
        processed_imagenet_classes.append((idx, name, processed, words))

    # Process each dataset class
    for dataset_idx, dataset_class in enumerate(dataset.classes):
        # Preprocess dataset class name
        dataset_class_lower = str(dataset_class).lower()
        dataset_class_processed = ''.join(c for c in dataset_class_lower if c.isalnum() or c.isspace())
        dataset_words = set(dataset_class_processed.split())

        best_match = None
        best_score = 0
        best_match_name = None

        # Try different matching strategies
        for imagenet_idx, imagenet_name, imagenet_processed, imagenet_words in processed_imagenet_classes:
            score = 0

            # Strategy 1: Exact match
            if dataset_class_processed == imagenet_processed:
                score = 100

            # Strategy 2: One is substring of the other
            elif dataset_class_processed in imagenet_processed:
                score = 50 + (len(dataset_class_processed) / len(imagenet_processed)) * 40
            elif imagenet_processed in dataset_class_processed:
                score = 50 + (len(imagenet_processed) / len(dataset_class_processed)) * 40

            # Strategy 3: Word overlap
            else:
                common_words = dataset_words.intersection(imagenet_words)
                if common_words:
                    word_score = len(common_words) / max(len(dataset_words), len(imagenet_words)) * 80
                    score = max(score, word_score)

            if score > best_score:
                best_score = score
                best_match = imagenet_idx
                best_match_name = imagenet_name

        # Use a threshold to ensure good matches
        if best_score >= 30:  # Adjust threshold as needed
            dataset_to_imagenet[dataset_idx] = best_match
        else:
            # If no good match, use the same index (assuming some alignment)
            dataset_to_imagenet[dataset_idx] = dataset_idx
            unmapped_classes.append((dataset_idx, dataset_class, best_match, best_match_name, best_score))

    # Count how many classes were mapped to different indices
    mapped_differently = 0
    for dataset_idx, imagenet_idx in dataset_to_imagenet.items():
        if dataset_idx != imagenet_idx:
            mapped_differently += 1

    print(f"Mapped {mapped_differently}/{len(dataset.classes)} classes to different ImageNet indices")

    # Print unmapped classes with their best (but insufficient) matches
    if unmapped_classes:
        print("\nClasses that couldn't be mapped (score < 30):")
        for idx, class_name, best_match, best_match_name, best_score in unmapped_classes:
            print(f"  Class {idx}: '{class_name}' → Best match was ImageNet {best_match}: '{best_match_name}' (score: {best_score:.1f})")

    return dataset_to_imagenet

# Add this function to map dataset classes based on folder names/IDs
def map_dataset_to_imagenet_by_folder_ids(dataset):
    """
    Map dataset classes to ImageNet classes based on folder IDs.

    Args:
        dataset: HFDataset with classes attribute

    Returns:
        Dictionary mapping dataset indices to ImageNet indices
    """
    # Try to get ImageNet class IDs from torchvision
    try:
        # Get the class folder names from ImageNet
        imagenet_model = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
        imagenet_class_names = torchvision.models.ResNet18_Weights.IMAGENET1K_V1.meta['categories']

        # Get the mapping of folder names to indices
        # This is typically in the format 'n01440764' -> 0, 'n01443537' -> 1, etc.
        # We'll need to extract these from the dataset class names if they're available

        print("\nAttempting to map dataset classes to ImageNet classes by folder IDs:")

        # Create mapping dictionary
        dataset_to_imagenet = {}

        # Check if the class names contain the folder IDs
        for dataset_idx, dataset_class in enumerate(dataset.classes):
            # Try to extract folder ID from class name
            # Assuming format like "n01440764" or "n01440764-tench"
            folder_id = None

            # Try to extract the folder ID using regex pattern for "nXXXXXXXX"
            import re
            match = re.search(r'n\d{8}', str(dataset_class))
            if match:
                folder_id = match.group(0)

            # If we couldn't extract from the class name, check if the class name itself is the folder ID
            if not folder_id and re.match(r'n\d{8}', str(dataset_class)):
                folder_id = str(dataset_class)

            # If we found a folder ID, try to find its index in ImageNet
            if folder_id:
                # In a real ImageNet dataset, we would look up the index directly
                # For now, we'll just use the dataset index as the ImageNet index
                # This assumes the classes are in the same order
                dataset_to_imagenet[dataset_idx] = dataset_idx

                if dataset_idx < 10:  # Print first 10 mappings
                    print(f"  Class {dataset_idx}: '{dataset_class}' → ImageNet index {dataset_idx}")
            else:
                if dataset_idx < 10:  # Print first 10 unmapped classes
                    print(f"  Class {dataset_idx}: '{dataset_class}' → No folder ID found")

        print(f"Mapped {len(dataset_to_imagenet)}/{len(dataset.classes)} classes to ImageNet classes by folder ID")

        # If we couldn't map any classes by folder ID, fall back to the original mapping
        if len(dataset_to_imagenet) == 0:
            print("No classes could be mapped by folder ID. Using original class indices.")
            # Just use the original indices (assuming they're already aligned)
            for i in range(len(dataset.classes)):
                dataset_to_imagenet[i] = i

        return dataset_to_imagenet

    except Exception as e:
        print(f"Error mapping classes by folder ID: {e}")
        print("Using original class indices.")
        # Just use the original indices (assuming they're already aligned)
        dataset_to_imagenet = {i: i for i in range(len(dataset.classes))}
        return dataset_to_imagenet

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

# --- Pruning and Fine-tuning ---
def main(filters_file_path=DEFAULT_FILTERS_FILE_PATH, epochs=num_epochs, batch_size=batch_size,
         learning_rate=learning_rate, no_show_plot=False):
    """Main function for pruning and fine-tuning.

    Args:
        filters_file_path: Path to the text file containing filters to prune
        epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Learning rate for optimizer
        no_show_plot: If True, don't display plots interactively (useful for batch processing)

    This implementation optimizes dataset loading by:
    1. Using a persistent cache directory to store datasets after first download
    2. Loading each dataset split only once and reusing it
    3. Applying transforms to the cached datasets

    This avoids redownloading the dataset on every run, significantly improving performance.
    """
    # Print training parameters
    print(f"Training parameters:")
    print(f"  Epochs: {epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  No show plot: {no_show_plot}")
    # Make sure we're using the GPU if available
    if torch.cuda.is_available():
        # Clear CUDA cache at the beginning to start with clean memory
        torch.cuda.empty_cache()
        print("Starting with clean GPU memory")

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

    # Print dataset structure information
    print_directory_contents(validation_dataset)

    # Map dataset classes to ImageNet classes using improved string matching
    class_mapping = map_dataset_to_imagenet_classes(validation_dataset)

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
    num_workers = min(0, os.cpu_count() or 1)  # Use up to 4 workers or available CPU cores

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

    # Skip evaluation before pruning to save time
    print("\nSkipping pre-pruning evaluation to save time...")

    # --- Prune filters ---
    print("\nPruning filters...")

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

                    # SIMPLER APPROACH: Just zero out the weights directly
                    # This is less likely to cause catastrophic accuracy drops
                    with torch.no_grad():
                        for filter_id in valid_filter_ids:
                            module.weight.data[filter_id].fill_(0)
                            if module.bias is not None:
                                module.bias.data[filter_id].fill_(0)

                    # Register a hook to keep these weights at zero during training
                    def make_hook(filter_indices):
                        def hook(grad):
                            for idx in filter_indices:
                                if idx < grad.shape[0]:
                                    grad[idx].fill_(0)
                            return grad
                        return hook

                    # Register the hook to zero gradients for pruned filters
                    module.weight.register_hook(make_hook(valid_filter_ids))
                    if module.bias is not None:
                        module.bias.register_hook(make_hook(valid_filter_ids))

                    pruned_count += len(valid_filter_ids)
                    print(f"Pruned {len(valid_filter_ids)} filters from Layer {layer_id} (kept {module.out_channels - len(valid_filter_ids)})")
            else:
                print(f"WARNING: Layer ID {layer_id} not found in model. Skipping {len(filter_ids)} filters.")

        print(f"\nPruning summary: Applied masks to {pruned_count} filters as specified")
        print("These filters will remain pruned during fine-tuning")

    # Apply direct pruning using the list provided at the top of the file
    apply_direct_pruning(model, filters_to_prune)

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

    # Create filter-specific checkpoint directory
    filter_basename = get_filter_file_basename(filters_file_path)
    filter_checkpoint_dir = os.path.join(checkpoint_dir, filter_basename)
    os.makedirs(filter_checkpoint_dir, exist_ok=True)
    print(f"Saving checkpoints to: {filter_checkpoint_dir}")
    best_val_acc = 0.0

    for epoch in range(epochs):
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

        # Save checkpoint after each epoch with filter name in the path
        checkpoint_path = os.path.join(filter_checkpoint_dir, f"checkpoint_epoch_{epoch+1}.pth")

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

        # Save best model with filter name in the path
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_path = os.path.join(filter_checkpoint_dir, f"{filter_basename}_best_model.pth")
            # Save best model to CPU for compatibility
            model_cpu = model.to('cpu')
            torch.save(model_cpu, best_model_path)
            # Move model back to original device
            model = model.to(device)
            print(f"New best model saved with validation accuracy: {val_acc:.2f}%")

        # Step the scheduler
        scheduler.step()

    # --- Save final model with filter name ---
    model.zero_grad()  # Clear gradients to reduce file size
    model_filename = f"{filter_basename}_resnet18_finetuned.pth"

    # Move model to CPU before saving to ensure compatibility
    model_cpu = model.to('cpu')
    torch.save(model_cpu, model_filename)
    print(f"Model saved to {model_filename}")

    # Move model back to original device
    model = model.to(device)

    # Also save final checkpoint with all training history and filter name
    final_checkpoint = os.path.join(filter_checkpoint_dir, f"{filter_basename}_final_checkpoint.pth")
    torch.save({
        'epochs_completed': epochs,
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

    # --- Plot accuracy with filter name in the filename ---
    plt.figure(figsize=(10, 6))
    plt.plot(train_acc_list, label="Train Accuracy")
    plt.plot(val_acc_list, label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title(f"Accuracy over Fine-Tuning Epochs ({filter_basename})")
    plt.legend()
    plt.grid()
    plt.savefig(f"{filter_basename}_accuracy_plot.png")

    # Either show the plot or close the figure based on the no_show_plot parameter
    if no_show_plot:
        plt.close()  # Close the figure without showing it
        print(f"Plot saved to {filter_basename}_accuracy_plot.png (not displayed)")
    else:
        plt.show()  # Show the plot interactively

if __name__ == "__main__":
    # Set PyTorch to use performance-optimized settings
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

    # Run with higher precision if available
    if torch.cuda.is_available():
        # Enable TF32 precision on Ampere GPUs (A100, RTX 30xx, etc.)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        print("TF32 precision enabled for compatible GPUs")

        # Set CUDA device to device 0
        torch.cuda.set_device(0)

    # Parse command-line arguments
    args = parse_args()
    print(f"Using filter file: {args.filter_file}")

    # Run the main function with the specified parameters
    main(
        filters_file_path=args.filter_file,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        no_show_plot=args.no_show_plot
    )