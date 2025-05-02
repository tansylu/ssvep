"""
Model Evaluation Module

This module provides functions to evaluate neural network models,
with specific focus on SSVEP-related evaluations and pruning analysis.
"""

import os
import sys
import argparse
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets

# Add project root to Python path if not already there
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from src.analysis.prune_filters import identify_filters_to_prune, create_pruned_model
except ImportError:
    print("Warning: Could not import pruning functions. Some evaluations may not work.")

def evaluate_model(model, test_loader, device='auto'):
    """
    Standard evaluation of model accuracy on classification test data.
    
    Args:
        model: PyTorch model to evaluate
        test_loader: DataLoader with test data
        device: Device to run evaluation on ('auto', 'cuda', or 'cpu')
    
    Returns:
        dict: Evaluation metrics
    """
    try:
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

def evaluate_model_inference_speed(model, test_loader, num_runs=10, device='auto'):
    """
    Evaluate model inference speed.
    
    Args:
        model: PyTorch model to evaluate
        test_loader: DataLoader with test data
        num_runs: Number of runs to average over
        device: Device to run evaluation on
        
    Returns:
        dict: Speed metrics
    """
    try:
        # Determine device
        if device == 'auto':
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            device = torch.device(device)
            
        model = model.to(device)
        model.eval()
        
        # Get a batch of data
        inputs, _ = next(iter(test_loader))
        inputs = inputs.to(device)
        batch_size = inputs.size(0)
        
        # Warmup
        with torch.no_grad():
            for _ in range(3):
                _ = model(inputs)
        
        # Measure time
        total_time = 0
        with torch.no_grad():
            for _ in range(num_runs):
                start = time.time()
                _ = model(inputs)
                torch.cuda.synchronize() if device.type == 'cuda' else None
                total_time += time.time() - start
                
        avg_time = total_time / num_runs
        throughput = batch_size / avg_time
        
        print(f"Inference Time: {avg_time*1000:.2f} ms per batch")
        print(f"Throughput: {throughput:.2f} images/sec")
        
        return {
            'batch_time': avg_time,
            'throughput': throughput,
            'batch_size': batch_size,
            'device': str(device)
        }
    except Exception as e:
        print(f"Error during speed evaluation: {e}")
        import traceback
        traceback.print_exc()
        return None

def evaluate_ssvep_model(model, flickering_data, frequencies, device='auto'):
    """
    Evaluate model performance on SSVEP frequency detection tasks.
    
    Args:
        model: PyTorch model to evaluate
        flickering_data: Dictionary mapping frequencies to data loaders with flickering stimuli
        frequencies: List of target frequencies to evaluate (e.g. [10, 12, 15])
        device: Device to run evaluation on
        
    Returns:
        dict: SSVEP-specific evaluation metrics
    """
    try:
        from collections import defaultdict
        
        # Determine device
        if device == 'auto':
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            device = torch.device(device)
            
        model = model.to(device)
        model.eval()
        
        # Store activations for each frequency
        frequency_responses = {}
        detection_accuracy = {}
        frequency_discrimination = {}
        
        # Hook to capture activations from model
        activations = {}
        def get_activation(name):
            def hook(model, input, output):
                activations[name] = output.detach().cpu()
            return hook
        
        # Register hooks for key layers
        hooks = []
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Conv2d):
                hooks.append(module.register_forward_hook(get_activation(name)))
        
        # Test each frequency
        for freq in frequencies:
            print(f"\nEvaluating frequency: {freq} Hz")
            
            # Get data loader for this frequency
            loader = flickering_data.get(freq)
            if not loader:
                print(f"No data found for {freq} Hz")
                continue
                
            freq_activations = defaultdict(list)
            correct_detections = 0
            total_samples = 0
            
            with torch.no_grad():
                for inputs, labels in loader:
                    # Assume inputs are flickering stimuli at frequency 'freq'
                    # and labels indicate the target frequency
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    
                    # Forward pass
                    outputs = model(inputs)
                    
                    # Get predicted frequencies (assuming model predicts freq indices)
                    _, predicted = outputs.max(1)
                    freq_indices = labels
                    
                    # Count correct detections
                    correct_detections += (predicted == freq_indices).sum().item()
                    total_samples += inputs.size(0)
                    
                    # Store layer activations for frequency analysis
                    for name, activation in activations.items():
                        freq_activations[name].append(activation)
            
            # Calculate detection accuracy
            accuracy = 100.0 * correct_detections / total_samples if total_samples > 0 else 0
            detection_accuracy[freq] = accuracy
            print(f"Detection accuracy for {freq} Hz: {accuracy:.2f}%")
            
            # Process activations for frequency analysis
            processed_activations = {}
            for layer_name, acts in freq_activations.items():
                # Combine activations from all batches
                all_acts = torch.cat(acts, dim=0)
                
                # Get temporal response by averaging across batch and spatial dimensions
                temporal_response = all_acts.mean(dim=[0, 2, 3])  # Average across batch, height, width
                
                # Store for spectral analysis
                processed_activations[layer_name] = temporal_response.numpy()
            
            # Store for overall results
            frequency_responses[freq] = processed_activations
            
        # Calculate frequency discrimination (how well model distinguishes between frequencies)
        for i, freq1 in enumerate(frequencies):
            for freq2 in frequencies[i+1:]:
                if freq1 in frequency_responses and freq2 in frequency_responses:
                    # Compare activations between different frequencies
                    # Higher values mean better discrimination
                    layer_diffs = []
                    
                    for layer_name in frequency_responses[freq1].keys():
                        acts1 = frequency_responses[freq1][layer_name]
                        acts2 = frequency_responses[freq2][layer_name]
                        
                        # Calculate spectral difference between responses
                        if len(acts1) == len(acts2):
                            diff = np.mean(np.abs(acts1 - acts2))
                            layer_diffs.append(diff)
                    
                    if layer_diffs:
                        discrimination = np.mean(layer_diffs)
                        frequency_discrimination[(freq1, freq2)] = discrimination
                        print(f"Discrimination between {freq1}Hz and {freq2}Hz: {discrimination:.4f}")
        
        # Cleanup hooks
        for hook in hooks:
            hook.remove()
            
        # Return metrics
        return {
            'detection_accuracy': detection_accuracy,
            'frequency_discrimination': frequency_discrimination,
            'frequency_responses': frequency_responses
        }
        
    except Exception as e:
        print(f"Error during SSVEP model evaluation: {e}")
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
        test_loader = DataLoader(
            test_dataset, batch_size=batch_size,
            shuffle=False, num_workers=2, pin_memory=True
        )
        
        return test_loader
    except Exception as e:
        print(f"Error loading test data: {e}")
        import traceback
        traceback.print_exc()
        return None

def create_flickering_data_loaders(base_images_dir, frequencies, frames_per_sequence=30, 
                                  batch_size=16, num_workers=2):
    """
    Create data loaders with flickering stimuli at different frequencies.
    
    Args:
        base_images_dir: Directory containing base images
        frequencies: List of frequencies to generate (Hz)
        frames_per_sequence: Number of frames in each flickering sequence
        batch_size: Batch size for data loaders
        
    Returns:
        dict: Mapping from frequency to data loader with flickering stimuli
    """
    class FlickeringDataset(Dataset):
        def __init__(self, image_dir, frequency, fps=60, frames=30, transform=None):
            """
            Dataset that creates flickering stimuli at specified frequency.
            
            Args:
                image_dir: Directory with base images
                frequency: Flicker frequency in Hz
                fps: Frames per second of output sequences
                frames: Number of frames in each sequence
                transform: Transforms to apply to images
            """
            self.image_paths = [os.path.join(image_dir, f) for f in os.listdir(image_dir) 
                               if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            self.frequency = frequency
            self.fps = fps
            self.frames = frames
            self.transform = transform
            
            # Calculate on/off pattern for this frequency
            period = fps / frequency  # frames per cycle
            self.on_off_pattern = [int(i % period < period/2) for i in range(frames)]
            
            print(f"Created dataset for {frequency}Hz with {len(self.image_paths)} base images")
            
        def __len__(self):
            return len(self.image_paths)
            
        def __getitem__(self, idx):
            # Load base image
            image_path = self.image_paths[idx]
            image = Image.open(image_path).convert('RGB')
            
            if self.transform:
                image = self.transform(image)
                
            # Create flickering sequence by modulating brightness
            sequence = []
            for i in range(self.frames):
                # Apply brightness based on on/off pattern
                brightness = 1.0 if self.on_off_pattern[i] else 0.2
                frame = image * brightness
                sequence.append(frame)
                
            # Stack frames into a sequence
            sequence_tensor = torch.stack(sequence)
            
            # For simplicity, use frequency index as label
            # In a real system you'd have proper labels
            label = torch.tensor(int(self.frequency), dtype=torch.long)
            
            return sequence_tensor, label
    
    # Standard image transforms
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # Create datasets and loaders for each frequency
    data_loaders = {}
    
    for freq in frequencies:
        dataset = FlickeringDataset(
            image_dir=base_images_dir,
            frequency=freq,
            transform=transform
        )
        
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers
        )
        
        data_loaders[freq] = loader
        
    return data_loaders

def plot_pruning_performance_curve(original_model, stats_df, test_loader, model_type="resnet18", 
                                  device="auto", output_path="pruning_curve.png"):
    """
    Plot network performance vs percentage of filters pruned.
    
    Args:
        original_model: Original PyTorch model
        stats_df: DataFrame with filter statistics
        test_loader: DataLoader with test data
        model_type: Type of model architecture
        device: Device for evaluation
        output_path: Path to save the plot
    """
    import numpy as np
    import matplotlib.pyplot as plt
    import time
    
    # Generate pruning percentages: 0%, 10%, 20%, ..., 100%
    percentages = np.arange(0, 1.1, 0.1)
    
    accuracies = []
    inference_times = []
    
    print("\n=== Testing Model Performance Across Pruning Rates ===")
    
    # Evaluate original model (0% pruned)
    print("\nEvaluating original model (0% pruned)...")
    original_metrics = evaluate_model(original_model, test_loader, device)
    
    if original_metrics is None:
        print("Error evaluating original model. Cannot create performance plot.")
        return
        
    accuracies.append(original_metrics['accuracy'])
    
    # Measure baseline inference time
    start_time = time.time()
    for _ in range(5):  # Average over multiple runs
        with torch.no_grad():
            for inputs, _ in test_loader:
                _ = original_model(inputs.to(device))
                break  # Just test one batch
    
    original_time = (time.time() - start_time) / 5
    inference_times.append(1.0)  # Normalized to 1.0
    
    # Test each pruning percentage (skip 0% as we already evaluated it)
    for percentage in percentages[1:]:
        print(f"\nTesting {percentage*100:.0f}% pruning rate...")
        
        # Identify filters to prune at this percentage
        filters_to_prune = identify_filters_to_prune(
            stats_df, 
            prune_percentage=percentage
        )
        
        # Create pruned model
        pruned_model = create_pruned_model(
            original_model, 
            filters_to_prune, 
            model_type
        )
        
        if pruned_model is None:
            print(f"Error creating pruned model at {percentage*100:.0f}% pruning. Skipping.")
            # Add placeholder to maintain alignment with percentages
            accuracies.append(None)
            inference_times.append(None)
            continue
        
        # Evaluate pruned model
        metrics = evaluate_model(pruned_model, test_loader, device)
        
        if metrics is None:
            print(f"Error evaluating pruned model at {percentage*100:.0f}% pruning. Skipping.")
            accuracies.append(None)
            inference_times.append(None)
            continue
        
        # Record accuracy
        accuracies.append(metrics['accuracy'])
        
        # Measure inference time
        start_time = time.time()
        for _ in range(5):
            with torch.no_grad():
                for inputs, _ in test_loader:
                    _ = pruned_model(inputs.to(device))
                    break
        
        pruned_time = (time.time() - start_time) / 5
        inference_times.append(original_time / pruned_time)  # Relative speedup
        
        print(f"Pruned {len(filters_to_prune)} filters ({percentage*100:.0f}%): " +
              f"Accuracy = {metrics['accuracy']:.2f}%, " +
              f"Speedup = {original_time/pruned_time:.2f}x")
    
    # Create the plot
    plt.figure(figsize=(10, 6))
    
    # Filter out None values
    valid_indices = [i for i, acc in enumerate(accuracies) if acc is not None]
    valid_percentages = [percentages[i]*100 for i in valid_indices]
    valid_accuracies = [accuracies[i] for i in valid_indices]
    
    # Plot accuracy curve
    plt.plot(valid_percentages, valid_accuracies, 'o-', color='blue', 
             linewidth=2, markersize=8, label='Accuracy')
    
    # Add reference line for original accuracy
    plt.axhline(y=original_metrics['accuracy'], color='green', linestyle='--', 
                label=f'Original Accuracy: {original_metrics["accuracy"]:.2f}%')
    
    # Add annotations
    for i, (x, y) in enumerate(zip(valid_percentages, valid_accuracies)):
        plt.annotate(f"{y:.1f}%", (x, y), textcoords="offset points", 
                    xytext=(0,10), ha='center')
    
    # Customize plot
    plt.xlabel('Percentage of Filters Pruned (%)')
    plt.ylabel('Accuracy (%)')
    plt.title('Network Performance vs. Filter Pruning Rate')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Set reasonable y-axis limits
    min_acc = max(0, min(valid_accuracies) - 5)
    max_acc = min(100, max(valid_accuracies) + 5)
    plt.ylim(min_acc, max_acc)
    
    # Save plot
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"\nPerformance curve plot saved to {output_path}")
    
    # Save data as CSV for future reference
    results_df = pd.DataFrame({
        'Pruning_Percentage': valid_percentages,
        'Accuracy': valid_accuracies
    })
    csv_path = output_path.replace('.png', '.csv')
    results_df.to_csv(csv_path, index=False)
    print(f"Performance data saved to {csv_path}")
    
    return valid_percentages, valid_accuracies

def compare_models(original_model, pruned_model, test_loader, device='auto', 
                  output_dir='model_comparison'):
    """
    Compare original and pruned models on the same test data.
    
    Args:
        original_model: Original PyTorch model
        pruned_model: Pruned PyTorch model
        test_loader: DataLoader with test data
        device: Device for evaluation
        output_dir: Directory to save comparison results
    
    Returns:
        dict: Comparison metrics
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Evaluate both models
    print("Evaluating original model...")
    original_metrics = evaluate_model(original_model, test_loader, device)
    
    print("\nEvaluating pruned model...")
    pruned_metrics = evaluate_model(pruned_model, test_loader, device)
    
    # Measure inference speed
    print("\nMeasuring original model inference speed...")
    original_speed = evaluate_model_inference_speed(original_model, test_loader, device=device)
    
    print("\nMeasuring pruned model inference speed...")
    pruned_speed = evaluate_model_inference_speed(pruned_model, test_loader, device=device)
    
    # Calculate changes
    if original_metrics and pruned_metrics:
        accuracy_change = pruned_metrics['accuracy'] - original_metrics['accuracy']
        loss_change = pruned_metrics['loss'] - original_metrics['loss']
        
        speedup = pruned_speed['throughput'] / original_speed['throughput'] if original_speed and pruned_speed else 0
        
        # Create a summary report
        report_path = os.path.join(output_dir, 'model_comparison_report.txt')
        with open(report_path, 'w') as f:
            f.write("=== Model Comparison Report ===\n\n")
            
            f.write("Accuracy:\n")
            f.write(f"  Original: {original_metrics['accuracy']:.2f}%\n")
            f.write(f"  Pruned: {pruned_metrics['accuracy']:.2f}%\n")
            f.write(f"  Change: {accuracy_change:.2f}%\n\n")
            
            f.write("Loss:\n")
            f.write(f"  Original: {original_metrics['loss']:.4f}\n")
            f.write(f"  Pruned: {pruned_metrics['loss']:.4f}\n")
            f.write(f"  Change: {loss_change:.4f}\n\n")
            
            f.write("Inference Speed:\n")
            f.write(f"  Original: {original_speed['throughput']:.2f} images/sec\n")
            f.write(f"  Pruned: {pruned_speed['throughput']:.2f} images/sec\n")
            f.write(f"  Speedup: {speedup:.2f}x\n\n")
            
            f.write("Resource Usage:\n")
            # Add code to calculate model size, memory usage, etc.
        
        print(f"Comparison report saved to {report_path}")
        
        # Create a comparison plot
        plt.figure(figsize=(10, 6))
        
        # Bar chart with accuracy and speed
        labels = ['Accuracy (%)', 'Throughput (imgs/sec)']
        original_values = [original_metrics['accuracy'], original_speed['throughput']]
        pruned_values = [pruned_metrics['accuracy'], pruned_speed['throughput']]
        
        x = np.arange(len(labels))
        width = 0.35
        
        plt.bar(x - width/2, original_values, width, label='Original Model')
        plt.bar(x + width/2, pruned_values, width, label='Pruned Model')
        
        plt.xlabel('Metric')
        plt.ylabel('Value')
        plt.title('Model Performance Comparison')
        plt.xticks(x, labels)
        plt.legend()
        
        plt.tight_layout()
        plot_path = os.path.join(output_dir, 'model_comparison.png')
        plt.savefig(plot_path)
        print(f"Comparison plot saved to {plot_path}")
        
        return {
            'original': original_metrics,
            'pruned': pruned_metrics,
            'original_speed': original_speed,
            'pruned_speed': pruned_speed,
            'accuracy_change': accuracy_change,
            'speedup': speedup
        }
    else:
        print("Could not complete comparison due to evaluation errors.")
        return None

def test_with_single_image(model, image_path, device='auto', model_type='resnet18'):
    """
    Test model on a single image.
    
    Args:
        model: PyTorch model or state_dict
        image_path: Path to image file
        device: Device to run inference on
        model_type: Model architecture if loading from state_dict
        
    Returns:
        dict: Prediction results
    """
    from PIL import Image
    import torchvision.models as models
    
    try:
        # Determine device
        if device == 'auto':
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            device = torch.device(device)
        
        # Check if model is a state dictionary
        is_state_dict = isinstance(model, dict) or hasattr(model, 'items')
        if is_state_dict:
            print("Model is a state dictionary, creating model instance...")
            
            # Create appropriate model instance based on model_type
            if model_type == 'resnet18':
                actual_model = models.resnet18(pretrained=False)
            elif model_type == 'resnet50':
                actual_model = models.resnet50(pretrained=False)
            else:
                raise ValueError(f"Unsupported model type: {model_type}")
                
            # Load the state dictionary
            actual_model.load_state_dict(model, strict=False)
            model = actual_model
            print(f"Created {model_type} model and loaded weights")
            
        # Now proceed with the model
        model = model.to(device)
        model.eval()
        
        # Load and preprocess image
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        image = Image.open(image_path).convert('RGB')
        img_tensor = transform(image).unsqueeze(0).to(device)
        
        # Run inference
        with torch.no_grad():
            start_time = time.time()
            outputs = model(img_tensor)
            inference_time = time.time() - start_time
            
            # Get predictions
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
            top5_prob, top5_idx = torch.topk(probabilities, 5)
        
        print(f"\n=== Single Image Test: {os.path.basename(image_path)} ===")
        print(f"Inference time: {inference_time*1000:.2f} ms")
        print("Top 5 predictions:")
        for i in range(5):
            print(f"  Class {top5_idx[i]}: {top5_prob[i]*100:.2f}%")
            
        return {
            'top_indices': top5_idx.cpu().numpy(),
            'top_probabilities': top5_prob.cpu().numpy(),
            'inference_time': inference_time
        }
    except Exception as e:
        print(f"Error testing image: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    # Example usage when run as script
    parser = argparse.ArgumentParser(description='Model Evaluation Tool')
    parser.add_argument('--model', type=str, required=True, help='Path to model file')
    parser.add_argument('--data', type=str, default=None, help='Path to test data directory')
    parser.add_argument('--image', type=str, default=None, help='Path to single test image')
    parser.add_argument('--device', type=str, default='auto', help='Device to use for evaluation')
    
    args = parser.parse_args()
    
    # Load the model
    try:
        model = torch.load(args.model, map_location='cpu')
        print(f"Loaded model from {args.model}")
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)
        
    # Test single image
    if args.image:
        test_with_single_image(model, args.image, args.device)
    
    # Test with dataset
    if args.data:
        test_loader = load_test_data(args.data)
        if test_loader:
            evaluate_model(model, test_loader, args.device)
            evaluate_model_inference_speed(model, test_loader, device=args.device)