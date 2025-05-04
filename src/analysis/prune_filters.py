import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import copy

'''
<<<<<<< HEAD
run with:
python src/analysis/prune_filters.py --model data/models/resnet18.pth --stats data/filter_stats.csv --model-type resnet18 --output pruned_output
=======
Filter Pruning Script for Neural Networks

Basic usage:
python src/analysis/prune_filters.py --model-name resnet18 --stats data/stats/filter_stats.csv --output pruned_output

For multiple pruning percentages in one run:
python src/analysis/prune_filters.py --percentages 0.1 0.2 0.3 0.4 0.5

For analysis only (without loading or modifying the model):
python src/analysis/prune_filters.py --analyze-only

The script will:
1. Look for filter statistics in data/stats/filter_stats.csv
2. Look for the model file in data/models/resnet18.pth
3. Identify filters to prune based on similarity scores
4. Create pruned versions of the model for each percentage
5. Save the pruned models and visualizations in separate directories
6. Generate evaluation metrics if test data is provided

Output Structure:
pruned_output/
├── pruned_10pct/
│   ├── pruned_model_10pct.pth
│   ├── pruned_filters.txt
│   ├── pruning_visualization.png
│   └── evaluation_results.txt (if test data provided)
├── pruned_20pct/
│   ├── ...
├── pruned_30pct/
│   ├── ...
└── evaluation_summary.csv (if test data provided)
>>>>>>> cf3ce56 (Initial commit with filter pruning functionality)
'''

# Add project root to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '../..'))
sys.path.insert(0, project_root)

<<<<<<< HEAD
# Import removed since we don't need database functions anymore
=======
try:
    from src.database import db
    from src.database.db_stats import get_filter_stats, export_filter_stats_to_csv
except ImportError:
    print("Warning: Could not import database modules. Some features may be limited.")
>>>>>>> cf3ce56 (Initial commit with filter pruning functionality)

def load_filter_stats(stats_file):
    """Load filter statistics from CSV file"""
    print(f"Loading filter statistics from {stats_file}")
    try:
        stats_df = pd.read_csv(stats_file)
        print(f"Loaded statistics for {len(stats_df)} filters")
        return stats_df
    except Exception as e:
        print(f"Error loading filter statistics: {e}")
        return None

def identify_filters_to_prune(stats_df, prune_percentage=0.3, min_score=None):
    """
    Identify filters to prune based on similarity scores.

    Args:
        stats_df: DataFrame containing filter statistics
        prune_percentage: Percentage of worst filters to prune (0-1)
        min_score: Minimum similarity score threshold (filters below this will be pruned)

    Returns:
        list: List of (layer_id, filter_id) tuples to prune
    """
    if stats_df is None or len(stats_df) == 0:
        print("No filter statistics available for pruning")
        return []
<<<<<<< HEAD
        
    # Make sure we have the right column names
    required_columns = ["Layer", "Filter", "Avg Similarity Score"]
    missing_columns = [col for col in required_columns if col not in stats_df.columns]
    
    if missing_columns:
        # Try alternative column names
        alt_columns = {
            "Layer": "layer_id", 
            "Filter": "filter_id", 
            "Avg Similarity Score": "avg_similarity_score"
        }
        
=======

    # Make sure we have the right column names
    required_columns = ["Layer", "Filter", "Avg Similarity Score"]
    missing_columns = [col for col in required_columns if col not in stats_df.columns]

    if missing_columns:
        # Try alternative column names
        alt_columns = {
            "Layer": "layer_id",
            "Filter": "filter_id",
            "Avg Similarity Score": "avg_similarity_score"
        }

>>>>>>> cf3ce56 (Initial commit with filter pruning functionality)
        for req_col, alt_col in alt_columns.items():
            if req_col in missing_columns and alt_col in stats_df.columns:
                stats_df = stats_df.rename(columns={alt_col: req_col})
                missing_columns.remove(req_col)
<<<<<<< HEAD
    
=======

>>>>>>> cf3ce56 (Initial commit with filter pruning functionality)
    if missing_columns:
        print(f"Error: Missing required columns in stats file: {missing_columns}")
        return []

    # Sort by similarity score (ascending)
    sorted_df = stats_df.sort_values('Avg Similarity Score', ascending=True)

    filters_to_prune = []

    # Prune based on minimum score threshold
    if min_score is not None:
        filters_to_prune = [(int(row['Layer']), int(row['Filter']))
                           for _, row in stats_df.iterrows()
                           if row['Avg Similarity Score'] < min_score]
        print(f"Identified {len(filters_to_prune)} filters with score below {min_score}")

    # Prune based on percentage
    else:
        num_to_prune = int(len(sorted_df) * prune_percentage)
        filters_to_prune = [(int(row['Layer']), int(row['Filter']))
                           for _, row in sorted_df.head(num_to_prune).iterrows()]
        print(f"Identified {len(filters_to_prune)} filters to prune ({prune_percentage*100:.1f}% of total)")

    return filters_to_prune

def map_layer_filter_to_pytorch(layer_id, filter_id, model_type="resnet18"):
    """
    Map a database layer_id and filter_id to actual PyTorch model layer name.
<<<<<<< HEAD
    Add range checks to prevent out of range filter indices.
    """
    if model_type == "resnet18":
        # Layer sizes in ResNet18
        layer_sizes = {
            "conv1": 64,
            "layer1.0.conv1": 64, "layer1.0.conv2": 64,
            "layer1.1.conv1": 64, "layer1.1.conv2": 64,
            "layer2.0.conv1": 128, "layer2.0.conv2": 128,
            "layer2.1.conv1": 128, "layer2.1.conv2": 128,
            "layer3.0.conv1": 256, "layer3.0.conv2": 256,
            "layer3.1.conv1": 256, "layer3.1.conv2": 256,
            "layer4.0.conv1": 512, "layer4.0.conv2": 512,
            "layer4.1.conv1": 512, "layer4.1.conv2": 512,
            "fc": 1000,
            "bn1": 64
        }
        
        layer_map = {
            0: "conv1",                # First conv layer (64 filters)
            1: "layer1.0.conv1",       # First block, first conv (64 filters)
            2: "layer1.0.conv2",       # First block, second conv (64 filters)
            3: "layer1.1.conv1",       # Second block in layer1, first conv
            4: "layer1.1.conv2",       # Second block in layer1, second conv
            5: "layer2.0.conv1",       # First block in layer2, first conv (128 filters)
            6: "layer2.0.conv2",       # First block in layer2, second conv (128 filters)
            # Note: Layer 7 is missing in your system map
            8: "layer2.1.conv1",       # Second block in layer2, first conv (128 filters)
            9: "layer2.1.conv2",       # Second block in layer2, second conv (128 filters)
            10: "layer3.0.conv1",      # First block in layer3, first conv (256 filters)
            11: "layer3.0.conv2",      # First block in layer3, second conv (256 filters)
            # Note: Layer 12 is missing in your system map
            13: "layer3.1.conv1",      # Second block in layer3, first conv (256 filters)
            14: "layer3.1.conv2",      # Second block in layer3, second conv (256 filters)
            15: "layer4.0.conv1",      # First block in layer4, first conv (512 filters)
            16: "layer4.0.conv2",      # First block in layer4, second conv (512 filters)
            # Note: Layer 17 is missing in your system map
            18: "layer4.1.conv1",      # Second block in layer4, first conv (512 filters)
            19: "layer4.1.conv2"       # Second block in layer4, second conv (512 filters)
        }
        
        if layer_id in layer_map:
            module_name = layer_map[layer_id]
            
            # Check if filter_id is within the valid range for this layer
            if module_name in layer_sizes and filter_id < layer_sizes[module_name]:
                return module_name, filter_id
            else:
                print(f"Warning: Filter index {filter_id} out of range for {module_name} (max: {layer_sizes.get(module_name, 'unknown')})")
                return None, None
=======
    """
    # Add more mappings based on your layer IDs
    if model_type == "resnet18":
        # Extended mapping for ResNet18
        layer_map = {
            1: "layer1.0.conv1",
            2: "layer1.0.conv2",
            3: "layer1.1.conv1",
            4: "layer1.1.conv2",
            5: "layer2.0.conv1",
            6: "layer2.0.conv2",
            7: "layer2.1.conv1",
            8: "layer2.1.conv2",
            9: "layer3.0.conv1",
            10: "layer3.0.conv2",
            11: "layer3.1.conv1",
            12: "layer3.1.conv2",
            13: "layer4.0.conv1",
            14: "layer4.0.conv2",
            15: "layer4.1.conv1",
            16: "layer4.1.conv2",
            17: "conv1",
            18: "fc",  # Fully connected layer
            19: "bn1",  # Batch norm layer
        }

        if layer_id in layer_map:
            return layer_map[layer_id], filter_id
>>>>>>> cf3ce56 (Initial commit with filter pruning functionality)
        else:
            print(f"Warning: Unknown layer_id {layer_id}")
            return None, None
    else:
        print(f"Warning: Unsupported model type {model_type}")
        return None, None

def load_model(model_path):
    """
    Load a PyTorch model from file.
<<<<<<< HEAD
    
    Args:
        model_path: Path to the PyTorch model file
        
=======

    Args:
        model_path: Path to the PyTorch model file

>>>>>>> cf3ce56 (Initial commit with filter pruning functionality)
    Returns:
        PyTorch model
    """
    try:
        import torch
        device = torch.device("cpu")
<<<<<<< HEAD
        
=======

>>>>>>> cf3ce56 (Initial commit with filter pruning functionality)
        # Try loading as a full model first
        try:
            model = torch.load(model_path, map_location=device)
            print(f"Loaded full model from {model_path}")
            return model
        except:
            # If that fails, try loading as a state dict
            print("Could not load as full model, trying as state dict...")
            state_dict = torch.load(model_path, map_location=device)
<<<<<<< HEAD
            
            # Try to determine model type and create model
            if any("resnet18" in k for k in state_dict.keys()):
                from torchvision.models import resnet18
                model = resnet18(weights=None)
=======

            # Try to determine model type and create model
            if any("resnet18" in k for k in state_dict.keys()):
                from torchvision.models import resnet18
                try:
                    # Try the new weights parameter style first
                    model = resnet18(weights=None)
                except TypeError:
                    # Fall back to old style if needed
                    model = resnet18(pretrained=False)
>>>>>>> cf3ce56 (Initial commit with filter pruning functionality)
                model.load_state_dict(state_dict)
                return model
            elif any("resnet50" in k for k in state_dict.keys()):
                from torchvision.models import resnet50
<<<<<<< HEAD
                model = resnet50(weights=None)
=======
                try:
                    # Try the new weights parameter style first
                    model = resnet50(weights=None)
                except TypeError:
                    # Fall back to old style if needed
                    model = resnet50(pretrained=False)
>>>>>>> cf3ce56 (Initial commit with filter pruning functionality)
                model.load_state_dict(state_dict)
                return model
            else:
                print("Could not determine model architecture from state dict.")
                return None
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def create_pruned_model(model, filters_to_prune, model_type="resnet18"):
    """Create a new model with pruned filters."""
    try:
        import torch
        import torch.nn as nn
        from torchvision import models
<<<<<<< HEAD
        
        # Check if model is a state dict
        is_state_dict = isinstance(model, dict) or hasattr(model, 'items')
        
=======

        # Check if model is a state dict
        is_state_dict = isinstance(model, dict) or hasattr(model, 'items')

>>>>>>> cf3ce56 (Initial commit with filter pruning functionality)
        if is_state_dict:
            print("Model is a state dictionary, creating pruned state dict")
            # For state dict approach
            state_dict = model if isinstance(model, dict) else model.state_dict()
<<<<<<< HEAD
            
=======

>>>>>>> cf3ce56 (Initial commit with filter pruning functionality)
            # Create a new state dict with detached tensors
            pruned_state_dict = {}
            for key, tensor in state_dict.items():
                # Clone and detach each tensor to avoid gradient issues
                if isinstance(tensor, torch.Tensor):
                    pruned_state_dict[key] = tensor.clone().detach()
                else:
                    pruned_state_dict[key] = tensor
<<<<<<< HEAD
            
            pruned_filters_count = 0
            
=======

            pruned_filters_count = 0

>>>>>>> cf3ce56 (Initial commit with filter pruning functionality)
            # Process each filter to prune
            for layer_id, filter_id in filters_to_prune:
                # Map database IDs to PyTorch layer name and filter index
                module_name, filter_index = map_layer_filter_to_pytorch(layer_id, filter_id, model_type)
<<<<<<< HEAD
                
                if module_name is None:
                    continue
                
                # For state dict, look for weights with this layer name
                weight_key = f"{module_name}.weight"
                bias_key = f"{module_name}.bias"
                
=======

                if module_name is None:
                    continue

                # For state dict, look for weights with this layer name
                weight_key = f"{module_name}.weight"
                bias_key = f"{module_name}.bias"

>>>>>>> cf3ce56 (Initial commit with filter pruning functionality)
                if weight_key in pruned_state_dict:
                    # Zero out filter weights if within range
                    if filter_index < pruned_state_dict[weight_key].shape[0]:
                        pruned_state_dict[weight_key][filter_index] = 0.0
                        pruned_filters_count += 1
                        print(f"Pruned filter {filter_index} in {weight_key}")
<<<<<<< HEAD
                        
=======

>>>>>>> cf3ce56 (Initial commit with filter pruning functionality)
                        # Zero out bias if it exists
                        if bias_key in pruned_state_dict:
                            pruned_state_dict[bias_key][filter_index] = 0.0
                    else:
                        print(f"Warning: Filter index {filter_index} out of range for {weight_key}")
                else:
                    print(f"Warning: Could not find {weight_key} in state dict")
<<<<<<< HEAD
            
=======

>>>>>>> cf3ce56 (Initial commit with filter pruning functionality)
            print(f"Total filters pruned in state dict: {pruned_filters_count}")

            # Create a new model with the pruned weights
            if model_type == "resnet18":
<<<<<<< HEAD
                new_model = models.resnet18(weights=None)
                new_model.load_state_dict(pruned_state_dict, strict=False)
                return new_model
            elif model_type == "resnet50":
                new_model = models.resnet50(weights=None)
=======
                try:
                    # Try the new weights parameter style first
                    new_model = models.resnet18(weights=None)
                except TypeError:
                    # Fall back to old style if needed
                    new_model = models.resnet18(pretrained=False)
                new_model.load_state_dict(pruned_state_dict, strict=False)
                return new_model
            elif model_type == "resnet50":
                try:
                    # Try the new weights parameter style first
                    new_model = models.resnet50(weights=None)
                except TypeError:
                    # Fall back to old style if needed
                    new_model = models.resnet50(pretrained=False)
>>>>>>> cf3ce56 (Initial commit with filter pruning functionality)
                new_model.load_state_dict(pruned_state_dict, strict=False)
                return new_model
            else:
                # Just return the state dict if we can't create a model
                print("Returning pruned state dictionary (no model created)")
                return pruned_state_dict
<<<<<<< HEAD
                
=======

>>>>>>> cf3ce56 (Initial commit with filter pruning functionality)
        else:
            # For full model approach
            print("Model is a full PyTorch model, creating pruned model")
            pruned_model = copy.deepcopy(model)
<<<<<<< HEAD
            
            # Dictionary to track which filters have been pruned in each module
            pruned_filters_by_module = {}
            
=======

            # Dictionary to track which filters have been pruned in each module
            pruned_filters_by_module = {}

>>>>>>> cf3ce56 (Initial commit with filter pruning functionality)
            # Process each filter to prune
            for layer_id, filter_id in filters_to_prune:
                # Map database IDs to PyTorch module and filter index
                module_name, filter_index = map_layer_filter_to_pytorch(layer_id, filter_id, model_type)
<<<<<<< HEAD
                
                if module_name is None:
                    continue
                    
=======

                if module_name is None:
                    continue

>>>>>>> cf3ce56 (Initial commit with filter pruning functionality)
                # Keep track of filters pruned in this module
                if module_name not in pruned_filters_by_module:
                    pruned_filters_by_module[module_name] = []
                pruned_filters_by_module[module_name].append(filter_index)
<<<<<<< HEAD
                
                print(f"Marking filter {filter_index} in module {module_name} for pruning")
                
=======

                print(f"Marking filter {filter_index} in module {module_name} for pruning")

>>>>>>> cf3ce56 (Initial commit with filter pruning functionality)
            # Actually perform the pruning
            for module_name, filter_indices in pruned_filters_by_module.items():
                try:
                    # Get the module by name
                    module = pruned_model
                    for part in module_name.split('.'):
                        if part.isdigit():
                            module = module[int(part)]
                        else:
                            module = getattr(module, part)
<<<<<<< HEAD
                    
=======

>>>>>>> cf3ce56 (Initial commit with filter pruning functionality)
                    # Zero out the filters
                    for filter_idx in filter_indices:
                        if hasattr(module, 'weight') and isinstance(module, nn.Conv2d):
                            if filter_idx < module.weight.shape[0]:
                                with torch.no_grad():
                                    module.weight.data[filter_idx, :, :, :] = 0.0
                                    if module.bias is not None and filter_idx < module.bias.shape[0]:
                                        module.bias.data[filter_idx] = 0.0
<<<<<<< HEAD
                                        
=======

>>>>>>> cf3ce56 (Initial commit with filter pruning functionality)
                            print(f"Pruned filter {filter_idx} in {module_name}")
                        else:
                            print(f"Warning: Module {module_name} is not a Conv2d or has no weight attribute")
                except Exception as e:
                    print(f"Error pruning {module_name}: {e}")
                    continue

            print(f"Total filters pruned in model: {sum(len(filters) for filters in pruned_filters_by_module.values())}")
            return pruned_model
<<<<<<< HEAD
        
=======

>>>>>>> cf3ce56 (Initial commit with filter pruning functionality)
    except Exception as e:
        print(f"Error creating pruned model: {e}")
        import traceback
        traceback.print_exc()
        return None

def save_pruned_model(model, output_path):
    """Save pruned model to a file."""
    try:
        import torch
        # Check if model is a state dict
        if isinstance(model, dict) or hasattr(model, 'items'):
            torch.save(model, output_path)
            print(f"Pruned model state dictionary saved to {output_path}")
        else:
            torch.save(model, output_path)
            print(f"Pruned model saved to {output_path}")
    except Exception as e:
        print(f"Error saving model: {e}")

<<<<<<< HEAD
def visualize_pruning(stats_df, filters_to_prune, output_path):
    """
    Create a visualization of which filters were pruned.
    
=======
def visualize_pruning(stats_df, filters_to_prune, output_path, prune_percentage=None):
    """
    Create a visualization of which filters were pruned.

>>>>>>> cf3ce56 (Initial commit with filter pruning functionality)
    Args:
        stats_df: DataFrame containing filter statistics
        filters_to_prune: List of (layer_id, filter_id) tuples that were pruned
        output_path: Path to save the visualization
<<<<<<< HEAD
=======
        prune_percentage: The percentage of filters pruned (for title and threshold calculation)
>>>>>>> cf3ce56 (Initial commit with filter pruning functionality)
    """
    if stats_df is None:
        print("No statistics available for visualization")
        return
<<<<<<< HEAD
        
    # Convert filters_to_prune to a set for faster lookup
    pruned_set = set(filters_to_prune)

    # Add a 'Pruned' column to the DataFrame
    stats_df['Pruned'] = stats_df.apply(
=======

    # Convert filters_to_prune to a set for faster lookup
    pruned_set = set(filters_to_prune)

    # Make a copy of the DataFrame to avoid modifying the original
    viz_df = stats_df.copy()

    # Add a 'Pruned' column to the DataFrame
    viz_df['Pruned'] = viz_df.apply(
>>>>>>> cf3ce56 (Initial commit with filter pruning functionality)
        lambda row: (int(row['Layer']), int(row['Filter'])) in pruned_set,
        axis=1
    )

    # Create a figure
    plt.figure(figsize=(10, 6))

    # Get unique layers
<<<<<<< HEAD
    unique_layers = sorted(stats_df['Layer'].unique())
=======
    unique_layers = sorted(viz_df['Layer'].unique())
>>>>>>> cf3ce56 (Initial commit with filter pruning functionality)
    num_layers = len(unique_layers)

    # Create a mapping from layer to y-position
    layer_to_y = {layer: i for i, layer in enumerate(unique_layers)}

    # Plot pruned filters in red, kept filters in blue
<<<<<<< HEAD
    pruned_df = stats_df[stats_df['Pruned']]
    kept_df = stats_df[~stats_df['Pruned']]
=======
    pruned_df = viz_df[viz_df['Pruned']]
    kept_df = viz_df[~viz_df['Pruned']]
>>>>>>> cf3ce56 (Initial commit with filter pruning functionality)

    # Add jitter to y-positions to avoid overlapping points
    def add_jitter(layer_series):
        return layer_series.map(layer_to_y) + np.random.uniform(-0.3, 0.3, size=len(layer_series))

    # Plot kept filters
    plt.scatter(
        kept_df['Avg Similarity Score'],
        add_jitter(kept_df['Layer']),
        color='blue', alpha=0.6, s=20, label='Kept Filters'
    )

    # Plot pruned filters
    plt.scatter(
        pruned_df['Avg Similarity Score'],
        add_jitter(pruned_df['Layer']),
        color='red', alpha=0.6, s=20, label='Pruned Filters'
    )

    # Add a vertical line at the pruning threshold
    if not pruned_df.empty:
<<<<<<< HEAD
=======
        # Get the threshold (highest similarity score among pruned filters)
>>>>>>> cf3ce56 (Initial commit with filter pruning functionality)
        threshold = pruned_df['Avg Similarity Score'].max()
        plt.axvline(x=threshold, color='red', linestyle='--',
                    label=f'Pruning Threshold: {threshold:.4f}')

    # Set y-ticks to layer numbers
    plt.yticks(range(num_layers), [f'Layer {layer}' for layer in unique_layers])

    # Add labels and title
    plt.xlabel('Similarity Score')
    plt.ylabel('Layer')
<<<<<<< HEAD
    plt.title('Filter Pruning Visualization')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
=======

    # Add percentage to title if provided
    if prune_percentage is not None:
        plt.title(f'Filter Pruning Visualization ({prune_percentage*100:.1f}% Pruned)')
    else:
        plt.title('Filter Pruning Visualization')

    plt.legend()
    plt.grid(True, alpha=0.3)

>>>>>>> cf3ce56 (Initial commit with filter pruning functionality)
    # Save the figure
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path)
<<<<<<< HEAD
    print(f"Pruning visualization saved to {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Prune filters based on similarity scores')
    parser.add_argument('--stats', type=str, required=True,
                        help='Path to the filter statistics CSV file')
    parser.add_argument('--model', type=str, required=True,
                        help='Path to the original model file')
    parser.add_argument('--output', type=str, default='pruned_model',
                        help='Output directory for the pruned model')
    parser.add_argument('--percentage', type=float, default=0.05,
                        help='Percentage of worst filters to prune (0-1)')
=======
    plt.close()  # Close the figure to free memory
    print(f"Pruning visualization saved to {output_path}")

def evaluate_model(model, test_loader, device='auto'):
    """
    Evaluate model accuracy on test data.

    Args:
        model: PyTorch model to evaluate
        test_loader: DataLoader with test data
        device: Device to run evaluation on ('auto', 'cuda', or 'cpu')

    Returns:
        dict: Evaluation metrics
    """
    try:
        import torch
        import torch.nn as nn

        # Determine device
        if device == 'auto':
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            device = torch.device(device)

        print(f"Evaluating model on {device}")

        model = model.to(device)
        model.eval()

        criterion = nn.CrossEntropyLoss()
        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # Forward pass
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                # Track metrics
                total_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                correct += predicted.eq(labels).sum().item()
                total += labels.size(0)

        # Calculate final metrics
        accuracy = 100.0 * correct / total
        avg_loss = total_loss / total

        print(f"Test Accuracy: {accuracy:.2f}%")
        print(f"Test Loss: {avg_loss:.4f}")

        return {
            'accuracy': accuracy,
            'loss': avg_loss,
            'correct': correct,
            'total': total,
        }
    except Exception as e:
        print(f"Error during model evaluation: {e}")
        import traceback
        traceback.print_exc()
        return None

def load_test_data(data_dir, batch_size=32):
    """
    Load test dataset for model evaluation.

    Args:
        data_dir: Directory containing test data
        batch_size: Batch size for data loading

    Returns:
        PyTorch DataLoader for test data
    """
    try:
        import torch
        from torchvision import datasets, transforms

        # Define data preprocessing
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        # Load dataset
        test_dataset = datasets.ImageFolder(data_dir, transform=transform)
        print(f"Loaded test dataset with {len(test_dataset)} images from {data_dir}")

        # Create data loader
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=batch_size,
            shuffle=False, num_workers=2, pin_memory=True
        )

        return test_loader
    except Exception as e:
        print(f"Error loading test data: {e}")
        import traceback
        traceback.print_exc()
        return None

def generate_stats_csv_from_db(output_path):
    """
    Generate filter statistics CSV file from the database.

    Args:
        output_path: Path to save the CSV file
    """
    try:
        # Check if there are stats in the database
        stats = get_filter_stats()
        if not stats:
            print("No filter statistics found in the database.")
            print("Make sure you've run the SSVEP analysis that populates the database first.")
            return False

        # Export to CSV
        print(f"Exporting {len(stats)} filter statistics to {output_path}")
        export_filter_stats_to_csv(output_path)
        print(f"Filter statistics exported to {output_path}")
        return True
    except Exception as e:
        print(f"Error generating statistics from database: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Prune filters based on similarity scores and evaluate performance')
    parser.add_argument('--stats', type=str, default='data/stats/filter_stats.csv',
                        help='Path to the filter statistics CSV file')
    parser.add_argument('--model-name', type=str, default='resnet18',
                        help='Name of the model (e.g., resnet18, resnet50)')
    parser.add_argument('--models-dir', type=str, default='data/models',
                        help='Directory containing model files')
    parser.add_argument('--output', type=str, default='pruned_model',
                        help='Output directory for the pruned model')
    parser.add_argument('--percentage', type=float, default=0.3,
                        help='Percentage of worst filters to prune (0-1)')
    parser.add_argument('--percentages', type=float, nargs='+',
                        help='Multiple percentages to prune (e.g., 0.1 0.2 0.3 for 10%%, 20%%, 30%%)')
>>>>>>> cf3ce56 (Initial commit with filter pruning functionality)
    parser.add_argument('--min-score', type=float, default=None,
                        help='Minimum similarity score threshold (filters below this will be pruned)')
    parser.add_argument('--model-type', type=str, default='resnet18',
                        help='Model architecture type')
<<<<<<< HEAD

    args = parser.parse_args()

    # Create output directory if it doesn't exist
    os.makedirs(args.output, exist_ok=True)

    # Load filter statistics directly from file
    stats_df = load_filter_stats(args.stats)
    
=======
    parser.add_argument('--test-data', type=str, default=None,
                        help='Directory containing test data for evaluation')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size for evaluation')
    parser.add_argument('--device', type=str, default='auto',
                        help="Device for evaluation: 'auto', 'cuda', or 'cpu'")
    parser.add_argument('--use-db', action='store_true',
                        help='Generate statistics CSV from database instead of loading from file')
    parser.add_argument('--analyze-only', action='store_true',
                        help='Only analyze which filters would be pruned without loading or modifying the model')

    args = parser.parse_args()

    # Create base output directory if it doesn't exist
    os.makedirs(args.output, exist_ok=True)

    # Get filter statistics
    stats_file = args.stats
    if args.use_db:
        print("Generating filter statistics from database...")
        # Ensure data/stats directory exists
        os.makedirs('data/stats', exist_ok=True)
        db_stats_path = os.path.join('data/stats', "filter_stats_from_db.csv")
        if generate_stats_csv_from_db(db_stats_path):
            stats_file = db_stats_path
        else:
            print("Failed to generate statistics from database, using file specified by --stats")

    stats_df = load_filter_stats(stats_file)

>>>>>>> cf3ce56 (Initial commit with filter pruning functionality)
    if stats_df is None:
        print("Error: Could not load filter statistics. Exiting.")
        return

<<<<<<< HEAD
    # Identify filters to prune
    filters_to_prune = identify_filters_to_prune(
        stats_df,
        prune_percentage=args.percentage,
        min_score=args.min_score
    )
    
    if not filters_to_prune:
        print("No filters identified for pruning. Exiting.")
        return

    # Load the original model
    original_model = load_model(args.model)
    
    if original_model is None:
        print("Error: Could not load original model. Exiting.")
        return

    # Create pruned model
    print("Creating pruned model...")
    pruned_model = create_pruned_model(original_model, filters_to_prune, args.model_type)
    
    if pruned_model is None:
        print("Error: Could not create pruned model. Exiting.")
        return
    
    # Save the pruned model
    pruned_model_path = os.path.join(args.output, 'pruned_model.pth')
    save_pruned_model(pruned_model, pruned_model_path)

    # Visualize pruning
    visualize_pruning(
        stats_df,
        filters_to_prune,
        os.path.join(args.output, 'pruning_visualization.png')
    )

    # Save the list of pruned filters
    with open(os.path.join(args.output, 'pruned_filters.txt'), 'w') as f:
        f.write("Layer,Filter,Score\n")
        for layer_id, filter_id in filters_to_prune:
            # Find the score for this filter
            filter_row = stats_df[(stats_df['Layer'] == layer_id) & (stats_df['Filter'] == filter_id)]
            if not filter_row.empty:
                score = filter_row['Avg Similarity Score'].values[0]
                f.write(f"{layer_id},{filter_id},{score:.6f}\n")
            else:
                f.write(f"{layer_id},{filter_id},unknown\n")

    print(f"\nPruned {len(filters_to_prune)} filters out of {len(stats_df)} total filters")
    print(f"Pruning details saved to {os.path.join(args.output, 'pruned_filters.txt')}")
    print(f"Pruned model saved to {pruned_model_path}")
    print(f"\nUse evaluate_models.py to test the performance of your pruned model")
=======
    # Determine which percentages to use
    percentages = []
    if args.percentages:
        percentages = sorted(args.percentages)  # Sort from smallest to largest
        print(f"Running pruning for multiple percentages: {[f'{p*100:.1f}%' for p in percentages]}")
    else:
        percentages = [args.percentage]  # Use the single percentage
        print(f"Running pruning for percentage: {args.percentage*100:.1f}%")

    # Process each pruning percentage
    for prune_percentage in percentages:
        print(f"\n{'='*50}")
        print(f"Processing pruning percentage: {prune_percentage*100:.1f}%")
        print(f"{'='*50}")

        # Create percentage-specific output directory
        percentage_dir = os.path.join(args.output, f"pruned_{int(prune_percentage*100)}pct")
        os.makedirs(percentage_dir, exist_ok=True)
        print(f"Output directory: {percentage_dir}")

        # Identify filters to prune for this percentage
        filters_to_prune = identify_filters_to_prune(
            stats_df,
            prune_percentage=prune_percentage,
            min_score=args.min_score
        )

        if not filters_to_prune:
            print(f"No filters identified for pruning at {prune_percentage*100:.1f}%. Skipping.")
            continue

        # Check if we're in analyze-only mode
        if args.analyze_only:
            print(f"\nAnalysis mode: Identified {len(filters_to_prune)} filters that would be pruned at {prune_percentage*100:.1f}%")

            # Save the list of filters that would be pruned
            pruned_filters_path = os.path.join(percentage_dir, 'filters_to_prune.txt')
            with open(pruned_filters_path, 'w') as f:
                f.write("Layer,Filter,Score\n")
                for layer_id, filter_id in filters_to_prune:
                    # Find the score for this filter
                    filter_row = stats_df[(stats_df['Layer'] == layer_id) & (stats_df['Filter'] == filter_id)]
                    if not filter_row.empty:
                        score = filter_row['Avg Similarity Score'].values[0]
                        f.write(f"{layer_id},{filter_id},{score:.6f}\n")
                    else:
                        f.write(f"{layer_id},{filter_id},unknown\n")

            print(f"List of filters that would be pruned saved to {pruned_filters_path}")

            # Visualize pruning
            vis_path = os.path.join(percentage_dir, 'pruning_analysis.png')
            visualize_pruning(
                stats_df,
                filters_to_prune,
                vis_path,
                prune_percentage
            )
            print(f"Pruning visualization saved to {vis_path}")
            continue  # Move to next percentage

        # Proceed with model loading and pruning for this percentage

        # Construct model path and load the original model (only once for the first percentage)
        if 'original_model' not in locals():
            model_filename = f"{args.model_name}.pth"
            model_path = os.path.join(args.models_dir, model_filename)

            print(f"Looking for model at: {model_path}")
            if not os.path.exists(model_path):
                print(f"Warning: Model file not found at {model_path}")
                # Try alternative extensions
                alt_extensions = ['.pt', '.ckpt', '.weights']
                for ext in alt_extensions:
                    alt_path = os.path.join(args.models_dir, f"{args.model_name}{ext}")
                    if os.path.exists(alt_path):
                        model_path = alt_path
                        print(f"Found alternative model file at {model_path}")
                        break
                else:
                    print("Error: Could not find model file. Exiting.")
                    return

            original_model = load_model(model_path)

            if original_model is None:
                print("Error: Could not load original model. Exiting.")
                return

        # Create pruned model for this percentage
        print(f"Creating pruned model with {prune_percentage*100:.1f}% pruning...")
        pruned_model = create_pruned_model(original_model, filters_to_prune, args.model_type)

        if pruned_model is None:
            print(f"Error: Could not create pruned model for {prune_percentage*100:.1f}%. Skipping.")
            continue

        # Save the pruned model
        pruned_model_path = os.path.join(percentage_dir, f'pruned_model_{int(prune_percentage*100)}pct.pth')
        save_pruned_model(pruned_model, pruned_model_path)
        print(f"Pruned model saved to {pruned_model_path}")

        # Visualize pruning
        vis_path = os.path.join(percentage_dir, 'pruning_visualization.png')
        visualize_pruning(
            stats_df,
            filters_to_prune,
            vis_path,
            prune_percentage
        )
        print(f"Pruning visualization saved to {vis_path}")

        # Save the list of pruned filters
        filters_path = os.path.join(percentage_dir, 'pruned_filters.txt')
        with open(filters_path, 'w') as f:
            f.write("Layer,Filter,Score\n")
            for layer_id, filter_id in filters_to_prune:
                # Find the score for this filter
                filter_row = stats_df[(stats_df['Layer'] == layer_id) & (stats_df['Filter'] == filter_id)]
                if not filter_row.empty:
                    score = filter_row['Avg Similarity Score'].values[0]
                    f.write(f"{layer_id},{filter_id},{score:.6f}\n")
                else:
                    f.write(f"{layer_id},{filter_id},unknown\n")
        print(f"List of pruned filters saved to {filters_path}")

    # Evaluate models if test data is provided
    if args.test_data and not args.analyze_only and 'original_model' in locals():
        print("\n=== Model Evaluation ===")
        # Load test data
        test_loader = load_test_data(args.test_data, args.batch_size)

        if test_loader:
            # Evaluate original model once
            print("\nEvaluating original model performance...")
            original_metrics = evaluate_model(original_model, test_loader, args.device)

            if original_metrics:
                # Create a summary file for all percentages
                summary_path = os.path.join(args.output, 'evaluation_summary.csv')
                with open(summary_path, 'w') as f:
                    f.write("Pruning Percentage,Accuracy,Loss,Accuracy Change,Loss Change,Pruned Filters\n")

                # Evaluate each pruned model
                for prune_percentage in percentages:
                    percentage_dir = os.path.join(args.output, f"pruned_{int(prune_percentage*100)}pct")
                    pruned_model_path = os.path.join(percentage_dir, f'pruned_model_{int(prune_percentage*100)}pct.pth')

                    if not os.path.exists(pruned_model_path):
                        print(f"Skipping evaluation for {prune_percentage*100:.1f}% - model not found")
                        continue

                    print(f"\nEvaluating pruned model ({prune_percentage*100:.1f}%)...")
                    pruned_model = load_model(pruned_model_path)

                    if pruned_model is None:
                        print(f"Error loading pruned model for {prune_percentage*100:.1f}%. Skipping evaluation.")
                        continue

                    pruned_metrics = evaluate_model(pruned_model, test_loader, args.device)

                    # Calculate changes and output comparison
                    if pruned_metrics:
                        acc_change = pruned_metrics['accuracy'] - original_metrics['accuracy']
                        loss_change = pruned_metrics['loss'] - original_metrics['loss']

                        # Identify filters pruned at this percentage
                        filters_to_prune = identify_filters_to_prune(
                            stats_df,
                            prune_percentage=prune_percentage,
                            min_score=args.min_score
                        )

                        # Save detailed evaluation results
                        results_path = os.path.join(percentage_dir, 'evaluation_results.txt')
                        with open(results_path, 'w') as f:
                            f.write(f"=== Pruning Evaluation Results ({prune_percentage*100:.1f}%) ===\n\n")
                            f.write(f"Original Model Accuracy: {original_metrics['accuracy']:.2f}%\n")
                            f.write(f"Pruned Model Accuracy: {pruned_metrics['accuracy']:.2f}%\n")
                            f.write(f"Accuracy Change: {acc_change:.2f}%\n\n")
                            f.write(f"Original Model Loss: {original_metrics['loss']:.4f}\n")
                            f.write(f"Pruned Model Loss: {pruned_metrics['loss']:.4f}\n")
                            f.write(f"Loss Change: {loss_change:.4f}\n\n")
                            f.write(f"Total Pruned Filters: {len(filters_to_prune)} out of {len(stats_df)}\n")

                        # Add to summary CSV
                        with open(summary_path, 'a') as f:
                            f.write(f"{prune_percentage*100:.1f},{pruned_metrics['accuracy']:.2f},{pruned_metrics['loss']:.4f},{acc_change:.2f},{loss_change:.4f},{len(filters_to_prune)}\n")

                        print(f"\nEvaluation summary for {prune_percentage*100:.1f}% pruning:")
                        print(f"  Original Accuracy: {original_metrics['accuracy']:.2f}%")
                        print(f"  Pruned Accuracy: {pruned_metrics['accuracy']:.2f}%")
                        print(f"  Accuracy Change: {acc_change:.2f}%")
                        print(f"  Detailed results saved to {results_path}")

                print(f"\nComplete evaluation summary saved to {summary_path}")

    print("\n=== Pruning Complete ===")
    if not args.analyze_only:
        print(f"Pruned models saved to {args.output}/")
    print("Run with --analyze-only to only analyze without creating models")
>>>>>>> cf3ce56 (Initial commit with filter pruning functionality)

if __name__ == "__main__":
    main()