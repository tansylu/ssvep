import os
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import matplotlib.pyplot as plt
from datasets import load_dataset

# --- CONFIG ---
csv_path = "data/filter_stats.csv"
pth_path = "data/models/resnet18.pth"
dataset_name = "Prisma-Multimodal/segmented-imagenet1k-subset"  # Hugging Face dataset name
batch_size = 64
num_epochs = 10
learning_rate = 5e-4  # Reduced learning rate for more stable training
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint_dir = "checkpoints"  # Directory to save intermediate models

# --- Load CSV ---
def load_filter_stats(stats_file):
    print(f"Loading filter statistics from {stats_file}")
    try:
        stats_df = pd.read_csv(stats_file)
        print(f"Loaded statistics for {len(stats_df)} filters")
        return stats_df
    except Exception as e:
        print(f"Error loading filter statistics: {e}")
        return None

def create_importance_scores_from_csv(stats_df, model):
    """Create a dictionary mapping layers to their filter importance scores"""
    required_columns = ["Layer", "Filter", "Avg Similarity Score"]
    alt_columns = {"Layer": "layer_id", "Filter": "filter_id", "Avg Similarity Score": "avg_similarity_score"}

    # Rename columns if needed
    for req_col, alt_col in alt_columns.items():
        if req_col not in stats_df.columns and alt_col in stats_df.columns:
            stats_df = stats_df.rename(columns={alt_col: req_col})

    if any(col not in stats_df.columns for col in required_columns):
        print("Missing required columns after rename.")
        return {}

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

    # Create a mapping table for your reference
    print("\nLayer ID Mapping (Your CSV Layer ID -> Model Layer):")
    print("CSV ID | Model Layer Name")
    print("-" * 30)
    for i in range(20):  # Assuming you have up to 20 layers
        matching_layers = [(name, m) for name, m, idx in conv_layers if isinstance(idx, int) and idx == i]
        if matching_layers:
            print(f"{i:6d} | {matching_layers[0][0]}")
        else:
            if i in [7, 12, 17]:  # Downsample layers
                print(f"{i:6d} | Downsample layer (skipped)")
            else:
                print(f"{i:6d} | Not found")

    # Create importance scores dictionary
    importance_dict = {}

    # Create a mapping from layer indices in CSV to actual model modules
    layer_mapping = {}

    # Count the number of actual layers (excluding skipped ones)
    actual_layer_count = sum(1 for _, _, idx in conv_layers if isinstance(idx, int))

    # Check if we need to adjust the layer indices
    max_layer_idx = stats_df["Layer"].max()
    if max_layer_idx >= actual_layer_count:
        print(f"\nWARNING: CSV contains layer indices up to {max_layer_idx}, but model only has {actual_layer_count} non-downsample Conv2d layers.")
        print("Attempting to create a mapping between CSV layer indices and model layers...")

        # Try to create a mapping based on layer sizes
        layer_sizes = {}
        for layer_idx, layer_df in stats_df.groupby("Layer"):
            max_filter_idx = layer_df["Filter"].max()
            layer_sizes[layer_idx] = max_filter_idx + 1  # +1 because indices are 0-based

        # Create a direct 1:1 mapping between CSV layer indices and model layer indices
        # This assumes that your CSV layer indices follow the same scheme as the model layer indices
        print("\nCreating direct 1:1 mapping between CSV layer indices and model layer indices")

        # Get all valid layer indices in the model
        valid_indices = [idx for _, _, idx in conv_layers if isinstance(idx, int)]

        # Map each CSV layer index directly to the corresponding model layer index if it exists
        for layer_idx in sorted(layer_sizes.keys()):
            if layer_idx in valid_indices:
                # Direct mapping - CSV layer index matches model layer index
                layer_mapping[layer_idx] = layer_idx

                # Find the corresponding layer name
                matching_layers = [(name, m) for name, m, idx in conv_layers if isinstance(idx, int) and idx == layer_idx]
                if matching_layers:
                    layer_name = matching_layers[0][0]
                    layer_size = matching_layers[0][1].out_channels
                    print(f"Mapped CSV layer {layer_idx} (size {layer_sizes[layer_idx]}) to model layer {layer_name} (index {layer_idx}, size {layer_size})")
            else:
                # If the layer index doesn't exist in the model (e.g., downsample layers), skip it
                print(f"Skipping CSV layer {layer_idx} - no matching model layer index")

    # Process each layer in the CSV
    for layer_idx, layer_df in stats_df.groupby("Layer"):
        # Determine which model layer to use
        model_idx = layer_mapping.get(layer_idx, layer_idx)

        # Find the corresponding layer in our custom indexed list
        matching_layers = [(name, m) for name, m, idx in conv_layers if isinstance(idx, int) and idx == model_idx]

        if matching_layers:
            name, module = matching_layers[0]
            print(f"\nProcessing CSV layer {layer_idx} -> model layer {model_idx} ({name})")

            # Create tensor of importance scores for this layer
            scores = torch.ones(module.out_channels)

            # Fill in the scores from the dataframe
            valid_filters = 0
            for _, row in layer_df.iterrows():
                filter_idx = int(row["Filter"])
                if filter_idx < module.out_channels:
                    # Lower similarity score means more redundant (directly use as importance)
                    scores[filter_idx] = float(row["Avg Similarity Score"])
                    valid_filters += 1

            print(f"Assigned scores to {valid_filters} filters out of {module.out_channels}")
            importance_dict[module] = scores
        else:
            print(f"WARNING: Layer index {layer_idx} (mapped to {model_idx}) not found in model layers.")

    return importance_dict

# We don't need the CSVImportance class since we're using unstructured pruning directly

# --- Load dataset using Hugging Face datasets ---
def load_dataset_from_huggingface(dataset_name_or_path, split="train", transform=None):
    """Load a dataset from Hugging Face datasets."""
    # Map split names to HF dataset splits
    split_mapping = {
        "train": "train",
        "val": "validation"
    }

    # Load the dataset
    hf_split = split_mapping.get(split, split)
    try:
        print(f"Loading dataset from Hugging Face Hub: {dataset_name_or_path}")
        dataset = load_dataset(dataset_name_or_path, split=hf_split)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        raise e

    # Create a PyTorch dataset
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

    return HFDataset(dataset, transform)

# --- Pruning and Fine-tuning ---
def main():
    # Load dataset to determine number of classes
    try:
        # Try to load a small portion of the validation dataset to get class info
        # Using validation split since we'll be using it as our training set
        temp_dataset = load_dataset_from_huggingface(dataset_name, split="validation")
        num_classes = len(temp_dataset.classes)
        print(f"Found {num_classes} classes in the dataset")
    except Exception as e:
        print(f"Warning: Could not determine number of classes from dataset: {e}")
        print("Using default 1000 classes for ImageNet.")
        num_classes = 1000

    # --- Load pretrained model ---
    model = torchvision.models.resnet18(weights=None)

    # Modify the final fully connected layer to match the number of classes in your dataset
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    # Load pretrained weights
    state_dict = torch.load(pth_path, map_location="cpu")
    model.load_state_dict(state_dict, strict=False)  # strict=False to ignore fc layer mismatch

    model.to(device)

    # --- Prune filters ---
    print("Pruning filters...")

    # Try to load filter statistics from CSV if available
    stats_df = load_filter_stats(csv_path) if os.path.exists(csv_path) else None

    # Instead of using structural pruning, let's implement unstructured pruning
    # This will set weights to zero instead of removing filters, preserving the architecture
    print("\nUsing unstructured pruning (weight masking) instead of structural pruning...")

    # Function to apply unstructured pruning to convolutional layers
    # Prune the bottom 10% of filters overall (across all layers)
    def apply_unstructured_pruning(model, importance_dict, pruning_ratio=0.1):  # Default to 10%
        pruned_filters_count = 0
        total_filters_count = 0

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
                conv_layers.append((f"skip_{i}", name, m))
            else:
                # This is a regular layer, assign the next available index that's not in skip_indices
                while idx in skip_indices:
                    idx += 1
                conv_layers.append((idx, name, m))
                idx += 1

        # Create a reverse mapping from module to index and name
        module_to_info = {module: (i, name) for i, name, module in conv_layers if isinstance(i, int)}

        # Print summary of modules with importance scores
        print("\nModules with importance scores:")
        print("Layer ID | Layer Name | Filters | Min Score | Max Score")
        print("-" * 70)

        # Collect all scores from all layers into a single list
        all_scores = []
        all_module_indices = []  # To keep track of which module and filter each score belongs to

        for module in importance_dict.keys():
            if module in module_to_info:
                idx, name = module_to_info[module]
                scores = importance_dict[module]
                min_score = scores.min().item()
                max_score = scores.max().item()
                filter_count = scores.shape[0]
                print(f"Layer {idx:2d} | {name:30s} | {filter_count:6d} | {min_score:.4f} | {max_score:.4f}")

                # Add scores to the global list
                for filter_idx, score in enumerate(scores):
                    all_scores.append(score.item())
                    all_module_indices.append((module, filter_idx))

                # Count total filters
                total_filters_count += filter_count

        # Calculate how many filters to prune overall
        num_to_prune_overall = int(total_filters_count * pruning_ratio)
        print(f"\nTotal filters: {total_filters_count}")
        print(f"Pruning {num_to_prune_overall} filters overall (bottom {pruning_ratio*100:.1f}%)")

        # Sort all scores to find the global threshold
        sorted_scores = sorted(all_scores)
        if num_to_prune_overall > 0 and num_to_prune_overall < len(sorted_scores):
            global_threshold = sorted_scores[num_to_prune_overall - 1]
        else:
            global_threshold = -1  # No pruning if num_to_prune_overall is invalid

        print(f"Global pruning threshold: {global_threshold:.4f} (filters with scores <= this value will be pruned)")

        # Create a dictionary to track which filters to prune in each module
        filters_to_prune = {module: [] for module in importance_dict.keys() if module in module_to_info}

        # Identify filters to prune based on the global threshold
        for i, score in enumerate(all_scores):
            if score <= global_threshold:
                module, filter_idx = all_module_indices[i]
                if module in filters_to_prune:
                    filters_to_prune[module].append(filter_idx)

        # Apply pruning to each module
        for module, filter_indices in filters_to_prune.items():
            if module in module_to_info and filter_indices:
                module_idx, name = module_to_info[module]
                scores = importance_dict[module]

                # Convert filter indices to tensor for easier manipulation
                indices = torch.tensor(filter_indices, dtype=torch.long)

                # Get pruned scores for reporting
                pruned_scores = scores[indices]
                min_score = scores.min().item()
                max_score = scores.max().item()
                avg_score = scores.mean().item()
                pruned_min = pruned_scores.min().item()
                pruned_max = pruned_scores.max().item()
                pruned_avg = pruned_scores.mean().item()

                print(f"\nPruning Layer {module_idx}: {name}")
                print(f"All filters - Min: {min_score:.4f}, Max: {max_score:.4f}, Avg: {avg_score:.4f}")
                print(f"Pruned filters - Min: {pruned_min:.4f}, Max: {pruned_max:.4f}, Avg: {pruned_avg:.4f}")

                # Format the indices list for better readability
                indices_list = indices.tolist()
                if len(indices_list) > 20:
                    # Show first 10 and last 10 if there are many indices
                    indices_str = str(indices_list[:10])[:-1] + ", ... " + str(indices_list[-10:])[1:]
                else:
                    indices_str = str(indices_list)
                print(f"Pruning {len(indices_list)} filters with indices: {indices_str}")

                # Apply pruning by zeroing out weights
                with torch.no_grad():
                    for idx in indices_list:
                        module.weight[idx].zero_()  # Set all weights for this filter to 0
                        if module.bias is not None:
                            module.bias[idx].zero_()  # Set bias to 0 as well
                        pruned_filters_count += 1

                print(f"Pruned {len(indices_list)} filters from Layer {module_idx}: {name} (kept {module.out_channels - len(indices_list)})")

        print(f"\nPruning summary: Zeroed out {pruned_filters_count} of {total_filters_count} filters ({pruned_filters_count/total_filters_count*100:.2f}%)")

    # Apply unstructured pruning
    if stats_df is not None and not stats_df.empty:
        # Create importance scores from CSV
        importance_dict = create_importance_scores_from_csv(stats_df, model)
        # Use default 10% pruning ratio (bottom 10% of filters)
        apply_unstructured_pruning(model, importance_dict)
    else:
        print("No CSV data available for pruning. Skipping pruning step.")

    # --- Dataloaders (upsample to 224x224) ---
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

    # Create datasets using Hugging Face datasets
    # Swap training and validation sets to have a larger training set
    print("Swapping training and validation sets to have a larger training set...")
    try:
        # Use the validation set as training set (it has 50,000 examples)
        train_dataset = load_dataset_from_huggingface(dataset_name, split="validation", transform=train_transform)
        print(f"Loaded {len(train_dataset)} training images (from validation split)")
    except Exception as e:
        print(f"Error loading training dataset: {e}")
        return

    try:
        # Use the training set as validation set (it has 10,000 examples)
        val_dataset = load_dataset_from_huggingface(dataset_name, split="train", transform=eval_transform)
        print(f"Loaded {len(val_dataset)} validation images (from train split)")
    except Exception as e:
        print(f"Error loading validation dataset: {e}")
        return

    # Ensure model's output layer matches the number of classes in the dataset
    if model.fc.out_features != len(train_dataset.classes):
        print(f"Adjusting model's FC layer from {model.fc.out_features} to {len(train_dataset.classes)} classes")
        model.fc = nn.Linear(model.fc.in_features, len(train_dataset.classes))
        model.to(device)
        num_classes = len(train_dataset.classes)
    else:
        num_classes = model.fc.out_features

    # Create DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    # --- Fine-tuning ---
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

    train_acc_list, val_acc_list = [], []
    
    # Create checkpoint directory if it doesn't exist
    os.makedirs(checkpoint_dir, exist_ok=True)
    best_val_acc = 0.0

    for epoch in range(num_epochs):
        model.train()
        correct, total, running_loss = 0, 0, 0.0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)

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
                inputs, targets = inputs.to(device), targets.to(device)

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
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_acc': train_acc,
            'val_acc': val_acc,
            'train_acc_history': train_acc_list,
            'val_acc_history': val_acc_list
        }, checkpoint_path)
        print(f"Checkpoint saved to {checkpoint_path}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_path = os.path.join(checkpoint_dir, "best_model.pth")
            torch.save(model, best_model_path)
            print(f"New best model saved with validation accuracy: {val_acc:.2f}%")

    # --- Save final model ---
    model.zero_grad()  # Clear gradients to reduce file size
    model_filename = "resnet18_segmented_imagenet_finetuned.pth"
    torch.save(model, model_filename)
    print(f"Model saved to {model_filename}")
    
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
        'best_val_acc': best_val_acc
    }, final_checkpoint)
    print(f"Final checkpoint saved to {final_checkpoint}")

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
    main()
