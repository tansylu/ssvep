import os
import argparse
import datetime
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from pruner.utils import HFImageNetDataset, get_imagenet_transforms, DEFAULT_DATASET_CACHE_DIR

def load_checkpoint(checkpoint_path, device):
    """
    Load a model checkpoint.

    Args:
        checkpoint_path: Path to the checkpoint file
        device: Device to load the model to

    Returns:
        model: The loaded model
        checkpoint_data: Dictionary containing checkpoint data
    """
    print(f"Loading checkpoint from {checkpoint_path}")

    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)

        # Check if it's a full model or just a checkpoint with state_dict
        if isinstance(checkpoint, torch.nn.Module):
            model = checkpoint
            checkpoint_data = {'epoch': 'unknown', 'train_acc': 0, 'val_acc': 0}
            print("Loaded complete model")
        else:
            # Import here to avoid circular imports
            import torchvision.models as models

            # Assume it's a ResNet18 model (modify as needed)
            model = models.resnet18(pretrained=False)
            model.fc = nn.Linear(model.fc.in_features, 1000)  # Adjust output size as needed

            # Load state dict
            model.load_state_dict(checkpoint['model_state_dict'])
            checkpoint_data = checkpoint
            print(f"Loaded model from checkpoint at epoch {checkpoint.get('epoch', 'unknown')}")

        model = model.to(device)
        model.eval()
        return model, checkpoint_data

    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        import traceback
        traceback.print_exc()
        return None, None
def prepare_hf_dataset(dataset_name, batch_size, cache_dir=None, split="test"):
    """
    Prepare HuggingFace dataset for evaluation.

    Args:
        dataset_name: Name of the HuggingFace dataset
        batch_size: Batch size for data loading
        cache_dir: Directory to cache the dataset
        split: Dataset split to use (test, validation, or train)

    Returns:
        data_loader: DataLoader for the dataset
        num_classes: Number of classes in the dataset
    """
    print(f"Loading dataset {dataset_name}")

    # Get transforms from utils
    transform = get_imagenet_transforms(train=False)

    # Load dataset
    try:
        # Ensure cache_dir exists
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
            print(f"Using cache directory: {cache_dir}")

        # Import here to avoid circular imports
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError("datasets library not installed. Please install it with 'pip install datasets'.")

        # Try loading the specified split first
        try:
            print(f"Trying to load {split} split...")
            dataset = load_dataset(dataset_name, split=split, cache_dir=cache_dir, trust_remote_code=True)
        except Exception as e:
            print(f"Could not load {split} split: {e}")

            # If test split was requested but not available, try validation
            if split == "test":
                print("Trying to load validation split instead...")
                try:
                    dataset = load_dataset(dataset_name, split="validation", cache_dir=cache_dir, trust_remote_code=True)
                except Exception as e:
                    print(f"Could not load validation split: {e}")
                    print("Falling back to train split...")
                    dataset = load_dataset(dataset_name, split="train", cache_dir=cache_dir, trust_remote_code=True)
            else:
                # If validation or train was requested but not available, try alternatives
                print("Trying alternative splits...")
                try:
                    dataset = load_dataset(dataset_name, split="test", cache_dir=cache_dir, trust_remote_code=True)
                except:
                    try:
                        dataset = load_dataset(dataset_name, split="validation", cache_dir=cache_dir, trust_remote_code=True)
                    except:
                        dataset = load_dataset(dataset_name, split="train", cache_dir=cache_dir, trust_remote_code=True)

        print(f"Successfully loaded dataset with {len(dataset)} examples from {split} split")
        print(f"Dataset features: {dataset.features}")

        # Print first example to understand structure
        print("\nFirst example:")
        first_example = dataset[0]
        for key, value in first_example.items():
            if hasattr(value, 'shape'):
                print(f"  {key}: {type(value)} with shape {value.shape}")
            else:
                print(f"  {key}: {type(value)}")

        # Create custom dataset with ImageNet mapping
        custom_dataset = HFImageNetDataset(dataset, transform=transform)
        print(f"Created dataset with {len(custom_dataset)} examples")
        print(f"Using label field: {custom_dataset.label_field}")
        print(f"Using image field: {custom_dataset.image_field}")

        # Create DataLoader
        data_loader = DataLoader(
            custom_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )

        return data_loader, 1000  # Always use 1000 classes for ImageNet

    except Exception as e:
        print(f"Error loading dataset: {e}")
        import traceback
        traceback.print_exc()
        return None, 0

def evaluate_model(model, data_loader, device, num_classes):
    """
    Evaluate model accuracy on a dataset.

    Args:
        model: PyTorch model to evaluate
        data_loader: DataLoader with evaluation data
        device: Device to run evaluation on
        num_classes: Number of classes in the dataset

    Returns:
        float: Accuracy percentage
    """
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating"):
            inputs = batch["pixel_values"].to(device)
            targets = batch["labels"].to(device)

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

            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    accuracy = 100. * correct / total
    return accuracy

def evaluate_checkpoints(checkpoint_dir, dataset_name=None, batch_size=None, cache_dir=None, split=None, skip_evaluation=False):
    """
    Load checkpoints and optionally evaluate them on a dataset.

    Args:
        checkpoint_dir: Directory containing checkpoint files
        dataset_name: Name of the HuggingFace dataset (optional if skip_evaluation=True)
        batch_size: Batch size for data loading (optional if skip_evaluation=True)
        cache_dir: Directory to cache the dataset (optional if skip_evaluation=True)
        split: Dataset split to use for evaluation (optional if skip_evaluation=True)
        skip_evaluation: If True, skip actual model evaluation and just extract stored accuracy values

    Returns:
        dict: Dictionary mapping checkpoint names to accuracy values and metadata
    """
    # Determine device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Find all checkpoint files
    checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pth')]
    checkpoint_files.sort()  # Sort to evaluate in order

    if not checkpoint_files:
        print(f"No checkpoint files found in {checkpoint_dir}")
        return {}, {}

    # Initialize data loader if evaluation is needed
    data_loader = None
    num_classes = 1000  # Default for ImageNet

    if not skip_evaluation:
        # Prepare dataset
        data_loader, num_classes = prepare_hf_dataset(dataset_name, batch_size, cache_dir, split)
        if data_loader is None:
            print("Warning: Could not load dataset. Falling back to using stored accuracy values.")
            skip_evaluation = True

    # Evaluate or extract data from each checkpoint
    results = {}
    detailed_results = {}

    for checkpoint_file in checkpoint_files:
        checkpoint_path = os.path.join(checkpoint_dir, checkpoint_file)

        try:
            # Load checkpoint data
            print(f"Loading {checkpoint_file}...")
            _, checkpoint_data = load_checkpoint(checkpoint_path, device)

            if checkpoint_data is None:
                print(f"Skipping {checkpoint_file} due to loading error")
                continue

            # Initialize metadata
            metadata = {
                'epoch': 'unknown',
                'train_acc': None,
                'val_acc': None,
                'pruning_stats': None
            }

            # Extract accuracy from checkpoint or evaluate
            if skip_evaluation:
                # Try to get accuracy from checkpoint data
                if 'val_acc' in checkpoint_data:
                    accuracy = checkpoint_data['val_acc']
                    print(f"Using stored validation accuracy for {checkpoint_file}: {accuracy:.2f}%")
                elif 'final_val_acc' in checkpoint_data:
                    accuracy = checkpoint_data['final_val_acc']
                    print(f"Using stored final validation accuracy for {checkpoint_file}: {accuracy:.2f}%")
                else:
                    # If no accuracy in checkpoint, use a placeholder
                    print(f"No accuracy data found in {checkpoint_file}, using placeholder")
                    accuracy = 0.0
            else:
                # Load the model for evaluation
                model, _ = load_checkpoint(checkpoint_path, device)
                if model is None:
                    print(f"Skipping {checkpoint_file} due to model loading error")
                    continue

                # Evaluate the model
                print(f"Evaluating {checkpoint_file}...")
                accuracy = evaluate_model(model, data_loader, device, num_classes)
                print(f"Evaluated accuracy for {checkpoint_file}: {accuracy:.2f}%")

                # Clear GPU memory
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            # Store basic accuracy result
            results[checkpoint_file] = accuracy

            # Extract additional metadata from checkpoint if available
            metadata['accuracy'] = accuracy

            if checkpoint_data:
                # Extract epoch information
                if 'epoch' in checkpoint_data:
                    metadata['epoch'] = checkpoint_data['epoch']

                # Extract training/validation accuracy if available
                if 'train_acc' in checkpoint_data:
                    metadata['train_acc'] = checkpoint_data['train_acc']
                if 'val_acc' in checkpoint_data:
                    metadata['val_acc'] = checkpoint_data['val_acc']

                # Extract pruning statistics if available
                if 'pruning_stats' in checkpoint_data:
                    metadata['pruning_stats'] = checkpoint_data['pruning_stats']

            # Store detailed results
            detailed_results[checkpoint_file] = metadata

            # Print available information
            print(f"Checkpoint: {checkpoint_file}")
            print(f"  Accuracy: {accuracy:.2f}%")
            if metadata['train_acc'] is not None:
                print(f"  Training accuracy: {metadata['train_acc']:.2f}%")
            if metadata['val_acc'] is not None:
                print(f"  Validation accuracy: {metadata['val_acc']:.2f}%")
            if metadata['epoch'] != 'unknown':
                print(f"  Epoch: {metadata['epoch']}")

        except Exception as e:
            print(f"Error processing {checkpoint_file}: {e}")
            import traceback
            traceback.print_exc()

    # Return both simple and detailed results
    return results, detailed_results

def plot_results(results, output_path=None):
    """
    Plot accuracy results for checkpoints, focusing on epoch-based comparison between
    pruning techniques.

    Args:
        results: Dictionary mapping checkpoint names to accuracy values
        output_path: Path to save the plot. If None, will save to data/retrain_evaluation/
    """
    # Set default output directory and create it if it doesn't exist
    output_dir = "data/retrain_evaluation"
    os.makedirs(output_dir, exist_ok=True)

    # Set default output path if not provided
    if output_path is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(output_dir, f"accuracy_comparison_{timestamp}.png")
    # Print all available checkpoints for debugging
    print("\nAll available checkpoints:")
    for checkpoint, accuracy in results.items():
        print(f"  {checkpoint}: {accuracy:.2f}%")

    # Extract epoch and accuracy data for regular and 10% pruned models
    regular_data = {}  # {epoch: accuracy}
    pruned_data = {}   # {epoch: accuracy}

    for checkpoint, accuracy in results.items():
        # Skip best models and final checkpoints
        if checkpoint.startswith("best_") or checkpoint.startswith("final_"):
            continue

        # Extract epoch number
        if "epoch" in checkpoint:
            try:
                # Print the checkpoint name for debugging
                print(f"Processing checkpoint: {checkpoint}")

                # Extract epoch number from the checkpoint filename
                # Handle different filename formats
                if "_10prcntg" in checkpoint:
                    # Format: checkpoint_epoch_X_10prcntg.pth
                    epoch_str = checkpoint.split("_epoch_")[1].split("_")[0]
                    epoch = int(epoch_str)
                    pruned_data[epoch] = accuracy
                    print(f"  Categorized as pruned model, epoch {epoch}")
                else:
                    # Format: checkpoint_epoch_X.pth
                    epoch_str = checkpoint.split("_epoch_")[1].split(".")[0]
                    epoch = int(epoch_str)
                    regular_data[epoch] = accuracy
                    print(f"  Categorized as regular model, epoch {epoch}")

            except Exception as e:
                # Print error for debugging
                print(f"Error processing {checkpoint}: {e}")
                # Skip if we can't parse the epoch
                continue

    # Create plot
    plt.figure(figsize=(12, 8))

    # Sort data by epoch
    regular_epochs = sorted(regular_data.keys())
    regular_accuracies = [regular_data[e] for e in regular_epochs]

    pruned_epochs = sorted(pruned_data.keys())
    pruned_accuracies = [pruned_data[e] for e in pruned_epochs]

    # Plot lines
    if regular_epochs:
        plt.plot(regular_epochs, regular_accuracies, 'o-', linewidth=2,
                 label="Only 7 Filters Pruned", color='blue', markersize=8)

    if pruned_epochs:
        plt.plot(pruned_epochs, pruned_accuracies, 's-', linewidth=2,
                 label="10% Pruned Model", color='red', markersize=8)

    # Set x-axis ticks to be integers from 1 to 10
    plt.xticks(range(1, 11))

    # Add labels and title
    plt.xlabel("Epoch", fontsize=14)
    plt.ylabel("Accuracy (%)", fontsize=14)
    plt.title("Impact of Different Pruning Levels on Accuracy", fontsize=16)
    plt.suptitle("Comparison of 7 Filters vs 10% Pruned Model Performance", fontsize=14, y=0.98)

    # Add grid and legend
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc='lower right', fontsize=12)

    # Set y-axis limits to better visualize the difference
    min_acc = min(min(regular_accuracies) if regular_accuracies else 100,
                  min(pruned_accuracies) if pruned_accuracies else 100)
    max_acc = max(max(regular_accuracies) if regular_accuracies else 0,
                  max(pruned_accuracies) if pruned_accuracies else 0)

    # Add some padding to the y-axis
    y_padding = (max_acc - min_acc) * 0.1
    plt.ylim(min_acc - y_padding, max_acc + y_padding)

    # Add text annotations for all accuracies
    if regular_epochs:
        # Annotate all regular model accuracies
        for i, (epoch, acc) in enumerate(zip(regular_epochs, regular_accuracies)):
            plt.annotate(f"{acc:.2f}%",
                        xy=(epoch, acc),
                        xytext=(5, 10), textcoords='offset points',
                        fontsize=10, color='blue')

    if pruned_epochs:
        # Annotate all pruned model accuracies
        for i, (epoch, acc) in enumerate(zip(pruned_epochs, pruned_accuracies)):
            plt.annotate(f"{acc:.2f}%",
                        xy=(epoch, acc),
                        xytext=(5, -15), textcoords='offset points',
                        fontsize=10, color='red')

    # Print a summary of the accuracies
    print("\nAccuracy Summary:")
    print("-" * 60)
    print(f"{'Epoch':<6} {'Only 7 Filters Pruned':<22} {'10% Pruned Model':<20} {'Difference':<10}")
    print("-" * 60)

    for epoch in range(1, 11):
        reg_acc = regular_data.get(epoch, None)
        pruned_acc = pruned_data.get(epoch, None)

        reg_str = f"{reg_acc:.2f}%" if reg_acc is not None else "N/A"
        pruned_str = f"{pruned_acc:.2f}%" if pruned_acc is not None else "N/A"

        diff = ""
        if reg_acc is not None and pruned_acc is not None:
            diff = f"{reg_acc - pruned_acc:.2f}%"

        print(f"{epoch:<6} {reg_str:<15} {pruned_str:<15} {diff:<10}")

    # Tight layout and save
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {output_path}")

    # Also save the data to a CSV for reference
    output_dir = os.path.dirname(output_path)
    csv_filename = os.path.basename(output_path).replace('.png', '_epoch_data.csv')
    csv_path = os.path.join(output_dir, csv_filename)

    with open(csv_path, 'w') as f:
        f.write("epoch,only7filters_accuracy,10percent_pruned_accuracy,difference\n")
        for epoch in range(1, 11):
            regular_acc = regular_data.get(epoch, "")
            pruned_acc = pruned_data.get(epoch, "")

            # Calculate difference if both values exist
            diff = ""
            if regular_acc and pruned_acc:
                diff = f"{regular_acc - pruned_acc:.4f}"

            # Format accuracy values
            if regular_acc:
                regular_acc = f"{regular_acc:.4f}"
            if pruned_acc:
                pruned_acc = f"{pruned_acc:.4f}"

            f.write(f"{epoch},{regular_acc},{pruned_acc},{diff}\n")
    print(f"Epoch data saved to CSV: {csv_path}")

    # Show plot
    plt.show()

def main():
    parser = argparse.ArgumentParser(description="Evaluate model checkpoints on HuggingFace dataset")
    parser.add_argument("--checkpoint-dir", type=str, required=True, help="Directory containing checkpoint files")
    parser.add_argument("--dataset", type=str, default="Prisma-Multimodal/segmented-imagenet1k-subset",
                        help="HuggingFace dataset name")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for evaluation")
    parser.add_argument("--cache-dir", type=str, default=DEFAULT_DATASET_CACHE_DIR, help="Cache directory for datasets")
    parser.add_argument("--output", type=str, default=None, help="Output path for accuracy plot")
    parser.add_argument("--split", type=str, default="test",
                        help="Dataset split to use (test, validation, or train)")
    parser.add_argument("--skip-evaluation", action="store_true",
                        help="Skip model evaluation and use stored accuracy values from checkpoints")

    args = parser.parse_args()

    # Use the output path provided by the user, or let plot_results use its default
    output_path = args.output
    if output_path is not None:
        print(f"Using user-specified output path: {output_path}")
    else:
        print("Output will be saved to data/retrain_evaluation/ directory")

    # Evaluate checkpoints
    results, detailed_results = evaluate_checkpoints(
        args.checkpoint_dir,
        args.dataset,
        args.batch_size,
        args.cache_dir,
        args.split,
        skip_evaluation=args.skip_evaluation
    )

    # Plot results
    if results:
        plot_results(results, output_path)

        # Save detailed results to CSV for future reference
        output_dir = os.path.dirname(output_path) if output_path else "data/retrain_evaluation"
        os.makedirs(output_dir, exist_ok=True)

        # Create filenames based on the output path or a default
        if output_path:
            base_filename = os.path.basename(output_path).replace('.png', '')
        else:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            base_filename = f"checkpoint_details_{timestamp}"

        csv_path = os.path.join(output_dir, f"{base_filename}_details.csv")

        try:
            with open(csv_path, 'w') as f:
                # Write header
                f.write("checkpoint,accuracy,epoch,train_acc,val_acc,pruning_type\n")

                # Write data
                for checkpoint, metadata in detailed_results.items():
                    epoch = metadata.get('epoch', 'unknown')
                    train_acc = metadata.get('train_acc', '')
                    val_acc = metadata.get('val_acc', '')
                    accuracy = metadata.get('accuracy', 0.0)

                    # Determine pruning type
                    if "10prcntg" in checkpoint:
                        pruning_type = "10_percent"
                    else:
                        pruning_type = "only7filters"

                    # Format values properly
                    if train_acc is not None:
                        train_acc = f"{train_acc:.4f}"
                    else:
                        train_acc = ""

                    if val_acc is not None:
                        val_acc = f"{val_acc:.4f}"
                    else:
                        val_acc = ""

                    f.write(f"{checkpoint},{accuracy:.4f},{epoch},{train_acc},{val_acc},{pruning_type}\n")

            print(f"Detailed results saved to CSV: {csv_path}")

            # Save pruning stats to a separate JSON file if available
            json_path = os.path.join(output_dir, f"{base_filename}_pruning_stats.json")
            has_pruning_stats = any(
                metadata.get('pruning_stats') is not None
                for metadata in detailed_results.values()
            )

            if has_pruning_stats:
                import json
                pruning_stats_by_checkpoint = {}

                for checkpoint, metadata in detailed_results.items():
                    if metadata.get('pruning_stats'):
                        pruning_stats_by_checkpoint[checkpoint] = metadata['pruning_stats']

                with open(json_path, 'w') as f:
                    json.dump(pruning_stats_by_checkpoint, f, indent=2)
                print(f"Pruning statistics saved to: {json_path}")

        except Exception as e:
            print(f"Error saving results to files: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("No results to plot")

if __name__ == "__main__":
    main()





