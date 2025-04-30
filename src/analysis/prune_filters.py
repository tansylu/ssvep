import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import copy

'''
run with:
python src/analysis/prune_filters.py --model data/models/resnet18.pth --stats data/filter_stats.csv --model-type resnet18 --output pruned_output
'''

# Add project root to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '../..'))
sys.path.insert(0, project_root)

try:
    from src.database import db
    from src.database.db_stats import get_filter_stats, export_filter_stats_to_csv
except ImportError:
    print("Warning: Could not import database modules. Some features may be limited.")

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
        
        for req_col, alt_col in alt_columns.items():
            if req_col in missing_columns and alt_col in stats_df.columns:
                stats_df = stats_df.rename(columns={alt_col: req_col})
                missing_columns.remove(req_col)
    
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
        else:
            print(f"Warning: Unknown layer_id {layer_id}")
            return None, None
    else:
        print(f"Warning: Unsupported model type {model_type}")
        return None, None

def load_model(model_path):
    """
    Load a PyTorch model from file.
    
    Args:
        model_path: Path to the PyTorch model file
        
    Returns:
        PyTorch model
    """
    try:
        import torch
        device = torch.device("cpu")
        
        # Try loading as a full model first
        try:
            model = torch.load(model_path, map_location=device)
            print(f"Loaded full model from {model_path}")
            return model
        except:
            # If that fails, try loading as a state dict
            print("Could not load as full model, trying as state dict...")
            state_dict = torch.load(model_path, map_location=device)
            
            # Try to determine model type and create model
            if any("resnet18" in k for k in state_dict.keys()):
                from torchvision.models import resnet18
                model = resnet18(pretrained=False)
                model.load_state_dict(state_dict)
                return model
            elif any("resnet50" in k for k in state_dict.keys()):
                from torchvision.models import resnet50
                model = resnet50(pretrained=False)
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
        
        # Check if model is a state dict
        is_state_dict = isinstance(model, dict) or hasattr(model, 'items')
        
        if is_state_dict:
            print("Model is a state dictionary, creating pruned state dict")
            # For state dict approach
            state_dict = model if isinstance(model, dict) else model.state_dict()
            
            # Create a new state dict with detached tensors
            pruned_state_dict = {}
            for key, tensor in state_dict.items():
                # Clone and detach each tensor to avoid gradient issues
                if isinstance(tensor, torch.Tensor):
                    pruned_state_dict[key] = tensor.clone().detach()
                else:
                    pruned_state_dict[key] = tensor
            
            pruned_filters_count = 0
            
            # Process each filter to prune
            for layer_id, filter_id in filters_to_prune:
                # Map database IDs to PyTorch layer name and filter index
                module_name, filter_index = map_layer_filter_to_pytorch(layer_id, filter_id, model_type)
                
                if module_name is None:
                    continue
                
                # For state dict, look for weights with this layer name
                weight_key = f"{module_name}.weight"
                bias_key = f"{module_name}.bias"
                
                if weight_key in pruned_state_dict:
                    # Zero out filter weights if within range
                    if filter_index < pruned_state_dict[weight_key].shape[0]:
                        pruned_state_dict[weight_key][filter_index] = 0.0
                        pruned_filters_count += 1
                        print(f"Pruned filter {filter_index} in {weight_key}")
                        
                        # Zero out bias if it exists
                        if bias_key in pruned_state_dict:
                            pruned_state_dict[bias_key][filter_index] = 0.0
                    else:
                        print(f"Warning: Filter index {filter_index} out of range for {weight_key}")
                else:
                    print(f"Warning: Could not find {weight_key} in state dict")
            
            print(f"Total filters pruned in state dict: {pruned_filters_count}")

            # Create a new model with the pruned weights
            if model_type == "resnet18":
                new_model = models.resnet18(pretrained=False)
                new_model.load_state_dict(pruned_state_dict, strict=False)
                return new_model
            elif model_type == "resnet50":
                new_model = models.resnet50(pretrained=False)
                new_model.load_state_dict(pruned_state_dict, strict=False)
                return new_model
            else:
                # Just return the state dict if we can't create a model
                print("Returning pruned state dictionary (no model created)")
                return pruned_state_dict
                
        else:
            # For full model approach
            print("Model is a full PyTorch model, creating pruned model")
            pruned_model = copy.deepcopy(model)
            
            # Dictionary to track which filters have been pruned in each module
            pruned_filters_by_module = {}
            
            # Process each filter to prune
            for layer_id, filter_id in filters_to_prune:
                # Map database IDs to PyTorch module and filter index
                module_name, filter_index = map_layer_filter_to_pytorch(layer_id, filter_id, model_type)
                
                if module_name is None:
                    continue
                    
                # Keep track of filters pruned in this module
                if module_name not in pruned_filters_by_module:
                    pruned_filters_by_module[module_name] = []
                pruned_filters_by_module[module_name].append(filter_index)
                
                print(f"Marking filter {filter_index} in module {module_name} for pruning")
                
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
                    
                    # Zero out the filters
                    for filter_idx in filter_indices:
                        if hasattr(module, 'weight') and isinstance(module, nn.Conv2d):
                            if filter_idx < module.weight.shape[0]:
                                with torch.no_grad():
                                    module.weight.data[filter_idx, :, :, :] = 0.0
                                    if module.bias is not None and filter_idx < module.bias.shape[0]:
                                        module.bias.data[filter_idx] = 0.0
                                        
                            print(f"Pruned filter {filter_idx} in {module_name}")
                        else:
                            print(f"Warning: Module {module_name} is not a Conv2d or has no weight attribute")
                except Exception as e:
                    print(f"Error pruning {module_name}: {e}")
                    continue

            print(f"Total filters pruned in model: {sum(len(filters) for filters in pruned_filters_by_module.values())}")
            return pruned_model
        
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

def visualize_pruning(stats_df, filters_to_prune, output_path):
    """
    Create a visualization of which filters were pruned.
    
    Args:
        stats_df: DataFrame containing filter statistics
        filters_to_prune: List of (layer_id, filter_id) tuples that were pruned
        output_path: Path to save the visualization
    """
    if stats_df is None:
        print("No statistics available for visualization")
        return
        
    # Convert filters_to_prune to a set for faster lookup
    pruned_set = set(filters_to_prune)

    # Add a 'Pruned' column to the DataFrame
    stats_df['Pruned'] = stats_df.apply(
        lambda row: (int(row['Layer']), int(row['Filter'])) in pruned_set,
        axis=1
    )

    # Create a figure
    plt.figure(figsize=(10, 6))

    # Get unique layers
    unique_layers = sorted(stats_df['Layer'].unique())
    num_layers = len(unique_layers)

    # Create a mapping from layer to y-position
    layer_to_y = {layer: i for i, layer in enumerate(unique_layers)}

    # Plot pruned filters in red, kept filters in blue
    pruned_df = stats_df[stats_df['Pruned']]
    kept_df = stats_df[~stats_df['Pruned']]

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
        threshold = pruned_df['Avg Similarity Score'].max()
        plt.axvline(x=threshold, color='red', linestyle='--',
                    label=f'Pruning Threshold: {threshold:.4f}')

    # Set y-ticks to layer numbers
    plt.yticks(range(num_layers), [f'Layer {layer}' for layer in unique_layers])

    # Add labels and title
    plt.xlabel('Similarity Score')
    plt.ylabel('Layer')
    plt.title('Filter Pruning Visualization')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save the figure
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path)
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
    parser.add_argument('--stats', type=str, default='filter_stats.csv',
                        help='Path to the filter statistics CSV file')
    parser.add_argument('--model', type=str, required=True,
                        help='Path to the original model file')
    parser.add_argument('--output', type=str, default='pruned_model',
                        help='Output directory for the pruned model')
    parser.add_argument('--percentage', type=float, default=0.3,
                        help='Percentage of worst filters to prune (0-1)')
    parser.add_argument('--min-score', type=float, default=None,
                        help='Minimum similarity score threshold (filters below this will be pruned)')
    parser.add_argument('--model-type', type=str, default='resnet18',
                        help='Model architecture type')
    parser.add_argument('--test-data', type=str, default=None,
                        help='Directory containing test data for evaluation')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size for evaluation')
    parser.add_argument('--device', type=str, default='auto',
                        help="Device for evaluation: 'auto', 'cuda', or 'cpu'")
    parser.add_argument('--use-db', action='store_true',
                        help='Generate statistics CSV from database instead of loading from file')

    args = parser.parse_args()

    # Create output directory if it doesn't exist
    os.makedirs(args.output, exist_ok=True)

    # Get filter statistics
    stats_file = args.stats
    if args.use_db:
        print("Generating filter statistics from database...")
        db_stats_path = os.path.join(args.output, "filter_stats_from_db.csv")
        if generate_stats_csv_from_db(db_stats_path):
            stats_file = db_stats_path
        else:
            print("Failed to generate statistics from database, using file specified by --stats")
    
    stats_df = load_filter_stats(stats_file)
    
    if stats_df is None:
        print("Error: Could not load filter statistics. Exiting.")
        return

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

    # Evaluate model if test data is provided
    if args.test_data:
        print("\n=== Model Evaluation ===")
        # Load test data
        test_loader = load_test_data(args.test_data, args.batch_size)
        
        if test_loader:
            # Evaluate original model
            print("\nEvaluating original model performance...")
            original_metrics = evaluate_model(original_model, test_loader, args.device)
            
            # Evaluate pruned model
            print("\nEvaluating pruned model performance...")
            pruned_metrics = evaluate_model(pruned_model, test_loader, args.device)
            
            # Calculate changes and output comparison
            if original_metrics and pruned_metrics:
                acc_change = pruned_metrics['accuracy'] - original_metrics['accuracy']
                loss_change = pruned_metrics['loss'] - original_metrics['loss']
                
                # Save evaluation results
                results_path = os.path.join(args.output, 'evaluation_results.txt')
                with open(results_path, 'w') as f:
                    f.write("=== Pruning Evaluation Results ===\n\n")
                    f.write(f"Original Model Accuracy: {original_metrics['accuracy']:.2f}%\n")
                    f.write(f"Pruned Model Accuracy: {pruned_metrics['accuracy']:.2f}%\n")
                    f.write(f"Accuracy Change: {acc_change:.2f}%\n\n")
                    f.write(f"Original Model Loss: {original_metrics['loss']:.4f}\n")
                    f.write(f"Pruned Model Loss: {pruned_metrics['loss']:.4f}\n")
                    f.write(f"Loss Change: {loss_change:.4f}\n\n")
                    f.write(f"Total Pruned Filters: {len(filters_to_prune)} out of {len(stats_df)}\n")
                
                print(f"\nEvaluation summary:")
                print(f"  Original Accuracy: {original_metrics['accuracy']:.2f}%")
                print(f"  Pruned Accuracy: {pruned_metrics['accuracy']:.2f}%")
                print(f"  Accuracy Change: {acc_change:.2f}%")
                print(f"\nDetailed results saved to {results_path}")
                
    print(f"\nPruned {len(filters_to_prune)} filters out of {len(stats_df)} total filters")
    print(f"Pruning details saved to {os.path.join(args.output, 'pruned_filters.txt')}")
    print(f"Pruned model saved to {pruned_model_path}")

if __name__ == "__main__":
    main()