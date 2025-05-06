import os
import sys
import argparse
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from PIL import Image
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets

# Add project root to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from src.analysis.prune_filters import identify_filters_to_prune, create_pruned_model
except ImportError:
    print("Warning: Could not import pruning functions. Some evaluations may not work.")

def load_model(model_path, device='cpu', model_type='resnet18'):
    """
    Load a model from file, handling both full models and state dictionaries.
    
    Args:
        model_path: Path to model file
        device: Device to load model to
        model_type: Model architecture for state dict loading
        
    Returns:
        PyTorch model
    """
    try:
        # Load model file with weights_only=False to handle legacy formats
        model_data = torch.load(model_path, map_location=device, weights_only=False)
        
        # Check if model is a state dictionary
        is_state_dict = isinstance(model_data, dict) or hasattr(model_data, 'items')
        
        if is_state_dict:
            print(f"Loading state dict into {model_type} architecture...")
            
            # Create model architecture
            if model_type == 'resnet18':
                model = models.resnet18(weights=None)
            elif model_type == 'resnet50':
                model = models.resnet50(weights=None)
            else:
                raise ValueError(f"Unsupported model type: {model_type}")
            
            # Load weights
            model.load_state_dict(model_data, strict=False)
        else:
            # Already a full model
            model = model_data
            
        model = model.to(device)
        model.eval()
        return model
    
    except Exception as e:
        print(f"Error loading model: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_with_single_image(model, image_path, device='cpu', top_k=5):
    """
    Test model on a single image.
    
    Args:
        model: PyTorch model
        image_path: Path to image file
        device: Device for inference
        top_k: Number of top predictions to return
        
    Returns:
        dict: Prediction results
    """
    try:
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
            top_prob, top_idx = torch.topk(probabilities, top_k)
        
        print(f"\n=== Single Image Test: {os.path.basename(image_path)} ===")
        print(f"Inference time: {inference_time*1000:.2f} ms")
        print(f"Top {top_k} predictions:")
        for i in range(min(top_k, len(top_prob))):
            print(f"  Class {top_idx[i]}: {top_prob[i]*100:.2f}%")
            
        return {
            'image': image_path,
            'top_indices': top_idx.cpu().numpy(),
            'top_probabilities': top_prob.cpu().numpy(),
            'inference_time': inference_time
        }
    except Exception as e:
        print(f"Error testing image: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_image_folder(model, folder_path, device='cpu', batch_size=32):
    """
    Test model accuracy on a folder of images.
    
    Args:
        model: PyTorch model
        folder_path: Path to folder with class subfolders of images
        device: Device for inference
        batch_size: Batch size for data loading
        
    Returns:
        dict: Evaluation metrics
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
        dataset = datasets.ImageFolder(folder_path, transform=transform)
        data_loader = DataLoader(
            dataset, batch_size=batch_size,
            shuffle=False, num_workers=2, pin_memory=True
        )
        
        print(f"Testing model on {len(dataset)} images from {folder_path}")
        
        # Evaluation metrics
        model.eval()
        correct = 0
        total = 0
        total_time = 0
        class_correct = {}
        class_total = {}
        
        # For per-class accuracy
        for class_idx, class_name in enumerate(dataset.classes):
            class_correct[class_idx] = 0
            class_total[class_idx] = 0
        
        # Evaluate
        with torch.no_grad():
            for batch_idx, (inputs, labels) in enumerate(data_loader):
                inputs, labels = inputs.to(device), labels.to(device)
                
                # Measure inference time
                start_time = time.time()
                outputs = model(inputs)
                batch_time = time.time() - start_time
                total_time += batch_time
                
                # Calculate accuracy
                _, predicted = outputs.max(1)
                batch_correct = predicted.eq(labels).sum().item()
                correct += batch_correct
                total += labels.size(0)
                
                # Track per-class accuracy
                for i in range(labels.size(0)):
                    label = labels[i].item()
                    class_total[label] += 1
                    if predicted[i].item() == label:
                        class_correct[label] += 1
                
                # Progress update
                if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == len(data_loader):
                    print(f"Batch {batch_idx+1}/{len(data_loader)}: " +
                          f"Accuracy so far: {100.0*correct/total:.2f}%")
        
        # Calculate metrics
        accuracy = 100.0 * correct / total
        avg_time = total_time / len(data_loader)
        throughput = total / total_time
        
        # Calculate per-class accuracy
        class_accuracies = {}
        for class_idx, class_name in enumerate(dataset.classes):
            if class_total[class_idx] > 0:
                class_accuracies[class_name] = 100.0 * class_correct[class_idx] / class_total[class_idx]
            else:
                class_accuracies[class_name] = 0.0
        
        # Print results
        print(f"\n=== Evaluation Results ===")
        print(f"Total images: {total}")
        print(f"Correct predictions: {correct}")
        print(f"Overall accuracy: {accuracy:.2f}%")
        print(f"Average batch time: {avg_time*1000:.2f} ms")
        print(f"Throughput: {throughput:.2f} images/sec")
        
        print("\nPer-class accuracy:")
        for class_name, class_acc in class_accuracies.items():
            print(f"  {class_name}: {class_acc:.2f}%")
        
        return {
            'accuracy': accuracy,
            'correct': correct,
            'total': total,
            'throughput': throughput,
            'class_accuracies': class_accuracies
        }
    except Exception as e:
        print(f"Error evaluating folder: {e}")
        import traceback
        traceback.print_exc()
        return None

def plot_pruning_performance(original_model, stats_df, test_folder, model_type="resnet18", 
                           device="cpu", output_path="pruning_curve.png"):
    """
    Plot classification accuracy vs percentage of filters pruned.
    
    Args:
        original_model: Original PyTorch model
        stats_df: DataFrame with filter statistics
        test_folder: Folder with test images in class subfolders
        model_type: Type of model architecture
        device: Device for evaluation
        output_path: Path to save the plot
    
    Returns:
        tuple: Lists of percentages and accuracies
    """
    try:
        # Generate pruning percentages from 0% to 100% in 10% steps
        percentages = np.arange(0, 1.1, 0.1)
        accuracies = []
        throughputs = []
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        
        print("\n=== Testing Model Performance Across Pruning Rates ===")
        
        # Evaluate original model (0% pruning)
        print("\nEvaluating original model (0% pruning)...")
        original_metrics = test_image_folder(original_model, test_folder, device)
        
        if original_metrics is None:
            print("Error evaluating original model. Cannot create performance plot.")
            return None, None
            
        accuracies.append(original_metrics['accuracy'])
        throughputs.append(original_metrics['throughput'])
        
        # Test each pruning percentage (skip 0% as we already tested it)
        for percentage in percentages[1:]:
            print(f"\n--- Testing {percentage*100:.0f}% pruning rate ---")
            
            # Identify filters to prune
            filters_to_prune = identify_filters_to_prune(
                stats_df, 
                prune_percentage=percentage
            )
            
            print(f"Identified {len(filters_to_prune)} filters to prune")
            
            # Create pruned model
            pruned_model = create_pruned_model(
                original_model, 
                filters_to_prune, 
                model_type
            )
            
            if pruned_model is None:
                print(f"Error creating pruned model at {percentage*100:.0f}% pruning. Skipping.")
                accuracies.append(None)
                throughputs.append(None)
                continue
            
            # Evaluate pruned model
            metrics = test_image_folder(pruned_model, test_folder, device)
            
            if metrics is None:
                print(f"Error evaluating pruned model at {percentage*100:.0f}% pruning. Skipping.")
                accuracies.append(None)
                throughputs.append(None)
                continue
            
            # Record metrics
            accuracies.append(metrics['accuracy'])
            throughputs.append(metrics['throughput'])
            
            print(f"Pruned {len(filters_to_prune)} filters ({percentage*100:.0f}%): " +
                  f"Accuracy = {metrics['accuracy']:.2f}%, " +
                  f"Throughput = {metrics['throughput']:.2f} images/sec")
        
        # Create the plots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
        
        # Filter out None values
        valid_indices = [i for i, acc in enumerate(accuracies) if acc is not None]
        valid_percentages = [percentages[i]*100 for i in valid_indices]
        valid_accuracies = [accuracies[i] for i in valid_indices]
        valid_throughputs = [throughputs[i] for i in valid_indices]
        
        # Plot accuracy curve (top plot)
        ax1.plot(valid_percentages, valid_accuracies, 'o-', color='blue', 
                linewidth=2, markersize=8)
        
        # Add reference line for original accuracy
        ax1.axhline(y=original_metrics['accuracy'], color='green', linestyle='--', 
                   label=f'Original Accuracy: {original_metrics["accuracy"]:.2f}%')
        
        # Add annotations
        for i, (x, y) in enumerate(zip(valid_percentages, valid_accuracies)):
            ax1.annotate(f"{y:.1f}%", (x, y), textcoords="offset points", 
                       xytext=(0,10), ha='center')
        
        # Customize first plot
        ax1.set_xlabel('Percentage of Filters Pruned (%)')
        ax1.set_ylabel('Accuracy (%)')
        ax1.set_title('Classification Accuracy vs. Filter Pruning Rate')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Set reasonable y-axis limits for accuracy
        min_acc = max(0, min(valid_accuracies) - 5)
        max_acc = min(100, max(valid_accuracies) + 5)
        ax1.set_ylim(min_acc, max_acc)
        
        # Plot throughput (bottom plot)
        ax2.plot(valid_percentages, valid_throughputs, 'o-', color='red', 
                linewidth=2, markersize=8)
        
        # Add reference line for original throughput
        ax2.axhline(y=original_metrics['throughput'], color='green', linestyle='--', 
                   label=f'Original: {original_metrics["throughput"]:.2f} imgs/sec')
        
        # Add annotations
        for i, (x, y) in enumerate(zip(valid_percentages, valid_throughputs)):
            ax2.annotate(f"{y:.1f}", (x, y), textcoords="offset points", 
                       xytext=(0,10), ha='center')
        
        # Customize second plot
        ax2.set_xlabel('Percentage of Filters Pruned (%)')
        ax2.set_ylabel('Throughput (images/sec)')
        ax2.set_title('Inference Speed vs. Filter Pruning Rate')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        # Save plot
        fig.tight_layout()
        plt.savefig(output_path)
        print(f"\nPerformance curves saved to {output_path}")
        
        # Save data as CSV for future reference
        results_df = pd.DataFrame({
            'Pruning_Percentage': valid_percentages,
            'Accuracy': valid_accuracies,
            'Throughput': valid_throughputs
        })
        csv_path = output_path.replace('.png', '.csv')
        results_df.to_csv(csv_path, index=False)
        print(f"Performance data saved to {csv_path}")
        
        return valid_percentages, valid_accuracies
        
    except Exception as e:
        print(f"Error plotting pruning performance: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def compare_two_models(model1_path, model2_path, test_folder, device='cpu', 
                     model1_type='resnet18', model2_type='resnet18',
                     output_dir='model_comparison'):
    """
    Compare two models on the same test data.
    
    Args:
        model1_path: Path to first model
        model2_path: Path to second model
        test_folder: Path to test data folder
        device: Device for evaluation
        model1_type: Architecture of first model
        model2_type: Architecture of second model
        output_dir: Directory to save results
    """
    try:
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Load models
        model1 = load_model(model1_path, device, model1_type)
        model2 = load_model(model2_path, device, model2_type)
        
        if model1 is None or model2 is None:
            print("Error loading models. Cannot compare.")
            return None
            
        # Evaluate models
        print("\nEvaluating first model...")
        model1_metrics = test_image_folder(model1, test_folder, device)
        
        print("\nEvaluating second model...")
        model2_metrics = test_image_folder(model2, test_folder, device)
        
        if model1_metrics is None or model2_metrics is None:
            print("Error evaluating models. Cannot compare.")
            return None
            
        # Calculate differences
        accuracy_diff = model2_metrics['accuracy'] - model1_metrics['accuracy']
        throughput_ratio = model2_metrics['throughput'] / model1_metrics['throughput']
        
        # Create comparison report
        report_path = os.path.join(output_dir, 'model_comparison.txt')
        with open(report_path, 'w') as f:
            f.write("=== Model Comparison Report ===\n\n")
            
            f.write(f"Model 1: {model1_path}\n")
            f.write(f"Model 2: {model2_path}\n\n")
            
            f.write("Accuracy:\n")
            f.write(f"  Model 1: {model1_metrics['accuracy']:.2f}%\n")
            f.write(f"  Model 2: {model2_metrics['accuracy']:.2f}%\n")
            f.write(f"  Difference: {accuracy_diff:.2f}%\n\n")
            
            f.write("Throughput:\n")
            f.write(f"  Model 1: {model1_metrics['throughput']:.2f} images/sec\n")
            f.write(f"  Model 2: {model2_metrics['throughput']:.2f} images/sec\n")
            f.write(f"  Ratio: {throughput_ratio:.2f}x\n\n")
            
            f.write("Per-class Accuracy Changes:\n")
            for class_name in model1_metrics['class_accuracies'].keys():
                acc1 = model1_metrics['class_accuracies'].get(class_name, 0)
                acc2 = model2_metrics['class_accuracies'].get(class_name, 0)
                diff = acc2 - acc1
                f.write(f"  {class_name}: {acc1:.2f}% → {acc2:.2f}% ({diff:+.2f}%)\n")
        
        print(f"Comparison report saved to {report_path}")
        
        # Create comparison chart
        plt.figure(figsize=(12, 10))
        
        # 1. Accuracy comparison
        plt.subplot(2, 1, 1)
        model_names = ['Model 1', 'Model 2']
        accuracies = [model1_metrics['accuracy'], model2_metrics['accuracy']]
        plt.bar(model_names, accuracies, color=['blue', 'green'])
        plt.ylabel('Accuracy (%)')
        plt.title('Classification Accuracy Comparison')
        
        # Add values on bars
        for i, v in enumerate(accuracies):
            plt.text(i, v/2, f"{v:.1f}%", ha='center', fontweight='bold', color='white')
        
        # 2. Throughput comparison
        plt.subplot(2, 1, 2)
        throughputs = [model1_metrics['throughput'], model2_metrics['throughput']]
        plt.bar(model_names, throughputs, color=['blue', 'green'])
        plt.ylabel('Throughput (images/sec)')
        plt.title('Inference Speed Comparison')
        
        # Add values on bars
        for i, v in enumerate(throughputs):
            plt.text(i, v/2, f"{v:.1f}", ha='center', fontweight='bold', color='white')
        
        plt.tight_layout()
        chart_path = os.path.join(output_dir, 'model_comparison_chart.png')
        plt.savefig(chart_path)
        print(f"Comparison chart saved to {chart_path}")
        
        return {
            'model1': model1_metrics,
            'model2': model2_metrics,
            'accuracy_difference': accuracy_diff,
            'throughput_ratio': throughput_ratio
        }
        
    except Exception as e:
        print(f"Error comparing models: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Model Evaluation Tool')
    parser.add_argument('--model', type=str, required=True, help='Path to model file')
    parser.add_argument('--model-type', type=str, default='resnet18', 
                      help='Model architecture type (resnet18, resnet50)')
    parser.add_argument('--data', type=str, default=None, help='Path to test data folder')
    parser.add_argument('--image', type=str, default=None, help='Path to single test image')
    parser.add_argument('--stats', type=str, default=None, help='Path to filter statistics CSV')
    parser.add_argument('--output', type=str, default='evaluation_results', help='Output directory')
    parser.add_argument('--compare', type=str, default=None, help='Path to second model for comparison')
    parser.add_argument('--plot-pruning', action='store_true', help='Plot pruning performance curve')
    parser.add_argument('--device', type=str, default='cpu', help='Device for evaluation: cpu or cuda')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Load model
    model = load_model(args.model, args.device, args.model_type)
    
    if model is None:
        sys.exit(1)
    
    # Test with single image
    if args.image:
        test_with_single_image(model, args.image, args.device)
    
    # Test with image folder
    if args.data:
        metrics = test_image_folder(model, args.data, args.device)
        
        # Save results to file
        if metrics:
            result_path = os.path.join(args.output, 'evaluation_results.txt')
            with open(result_path, 'w') as f:
                f.write("=== Model Evaluation Results ===\n\n")
                f.write(f"Model: {args.model}\n")
                f.write(f"Test data: {args.data}\n\n")
                f.write(f"Accuracy: {metrics['accuracy']:.2f}%\n")
                f.write(f"Throughput: {metrics['throughput']:.2f} images/sec\n\n")
                
                f.write("Per-class accuracy:\n")
                for class_name, acc in metrics['class_accuracies'].items():
                    f.write(f"  {class_name}: {acc:.2f}%\n")
                    
            print(f"Evaluation results saved to {result_path}")
    
    # Compare with another model
    if args.compare and args.data:
        compare_two_models(
            args.model, args.compare, 
            args.data, args.device, 
            args.model_type, args.model_type,
            os.path.join(args.output, 'comparison')
        )
    
    # Plot pruning performance curve
    if args.plot_pruning and args.data and args.stats:
        # Load filter statistics
        try:
            stats_df = pd.read_csv(args.stats)
            print(f"Loaded filter statistics from {args.stats}")
            
            plot_path = os.path.join(args.output, 'pruning_performance.png')
            plot_pruning_performance(
                model, 
                stats_df, 
                args.data, 
                args.model_type, 
                args.device, 
                plot_path
            )
        except Exception as e:
            print(f"Error loading statistics or plotting: {e}")
