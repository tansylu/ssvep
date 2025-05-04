import os
import argparse
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as transforms
from evaluate_models import load_model, test_with_single_image

def classify_images_with_model(model_path, images_dir, model_type='resnet18', device='cpu'):
    """
    Classify all images in a directory using the specified model.
    
    Args:
        model_path: Path to model file
        images_dir: Directory containing images
        model_type: Model architecture type
        device: Device for inference
    """
    # Load model
    print(f"Loading model from {model_path}...")
    model = load_model(model_path, device, model_type)
    
    if model is None:
        print("Error loading model. Exiting.")
        return
    
    # Get all image files
    image_files = [f for f in os.listdir(images_dir) 
                  if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    if not image_files:
        print(f"No images found in {images_dir}")
        return
    
    print(f"Found {len(image_files)} images to classify")
    
    # Process each image
    results = []
    for img_file in image_files:
        img_path = os.path.join(images_dir, img_file)
        print(f"\nClassifying {img_file}...")
        
        # Test with single image
        result = test_with_single_image(model, img_path, device)
        if result:
            results.append(result)
    
    # Calculate average inference time
    if results:
        avg_time = sum(r['inference_time'] for r in results) / len(results)
        print(f"\n=== Summary ===")
        print(f"Model: {model_path}")
        print(f"Images processed: {len(results)}")
        print(f"Average inference time: {avg_time*1000:.2f} ms")
    
    return results

def save_results_to_csv(results, output_path):
    """
    Save classification results to a CSV file.
    
    Args:
        results: Dictionary containing results for original and pruned models
        output_path: Path to save the CSV file
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Extract data
    original_results = results.get('original_results', [])
    pruned_results = results.get('pruned_results', [])
    
    if not original_results or not pruned_results:
        print("No results to save.")
        return
    
    # Create DataFrame
    data = []
    for i, (orig, pruned) in enumerate(zip(original_results, pruned_results)):
        image_name = os.path.basename(orig['image'])
        orig_time = orig['inference_time'] * 1000  # Convert to ms
        pruned_time = pruned['inference_time'] * 1000  # Convert to ms
        speedup = orig_time / pruned_time if pruned_time > 0 else 0
        
        # Check if top predictions match
        top1_match = orig['top_indices'][0] == pruned['top_indices'][0]
        
        # Calculate top-5 accuracy
        orig_top5 = set(orig['top_indices'][:5])
        pruned_top5 = set(pruned['top_indices'][:5])
        top5_match = len(orig_top5.intersection(pruned_top5)) / 5.0
        
        # Format top-5 predictions for both models
        orig_top5_str = ', '.join([f"{idx}({prob*100:.1f}%)" for idx, prob in 
                                  zip(orig['top_indices'][:5], orig['top_probabilities'][:5])])
        pruned_top5_str = ', '.join([f"{idx}({prob*100:.1f}%)" for idx, prob in 
                                    zip(pruned['top_indices'][:5], pruned['top_probabilities'][:5])])
        
        data.append({
            'Image': image_name,
            'Original_Time_ms': orig_time,
            'Pruned_Time_ms': pruned_time,
            'Speedup': speedup,
            'Original_Top5': orig_top5_str,
            'Pruned_Top5': pruned_top5_str,
            'Top1_Match': top1_match,
            'Top5_Overlap': top5_match
        })
    
    # Create DataFrame and save to CSV
    df = pd.DataFrame(data)
    df.to_csv(output_path, index=False)
    print(f"Detailed results saved to {output_path}")
    
    # Create a summary DataFrame with statistics
    top1_matches = sum(df['Top1_Match'])
    avg_top5_overlap = df['Top5_Overlap'].mean()
    
    summary_data = {
        'Total_Images': len(df),
        'Top1_Matches': top1_matches,
        'Top1_Match_Percentage': top1_matches / len(df) * 100,
        'Top1_Differences': len(df) - top1_matches,
        'Top1_Difference_Percentage': (len(df) - top1_matches) / len(df) * 100,
        'Avg_Top5_Overlap': avg_top5_overlap * 100,
        'Avg_Speedup': df['Speedup'].mean(),
        'Min_Speedup': df['Speedup'].min(),
        'Max_Speedup': df['Speedup'].max()
    }
    
    summary_df = pd.DataFrame([summary_data])
    summary_path = output_path.replace('.csv', '_summary.csv')
    summary_df.to_csv(summary_path, index=False)
    print(f"Summary statistics saved to {summary_path}")
    
    return df

def create_comparison_plots(results, output_dir):
    """
    Create plots comparing original and pruned model performance.
    
    Args:
        results: Dictionary containing results for original and pruned models
        output_dir: Directory to save the plots
    """
    # Create directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract data
    original_results = results.get('original_results', [])
    pruned_results = results.get('pruned_results', [])
    
    if not original_results or not pruned_results:
        print("No results to plot.")
        return
    
    # Extract inference times
    orig_times = [r['inference_time'] * 1000 for r in original_results]  # Convert to ms
    pruned_times = [r['inference_time'] * 1000 for r in pruned_results]  # Convert to ms
    
    # Calculate statistics
    orig_avg = np.mean(orig_times)
    pruned_avg = np.mean(pruned_times)
    speedup = orig_avg / pruned_avg if pruned_avg > 0 else 0
    
    # 1. Bar chart comparing average inference times
    plt.figure(figsize=(10, 6))
    plt.bar(['Original Model', 'Pruned Model'], [orig_avg, pruned_avg], color=['blue', 'green'])
    plt.ylabel('Average Inference Time (ms)')
    plt.title(f'Average Inference Time Comparison\nSpeedup: {speedup:.2f}x')
    
    # Add values on bars
    plt.text(0, orig_avg/2, f"{orig_avg:.2f} ms", ha='center', va='center', color='white', fontweight='bold')
    plt.text(1, pruned_avg/2, f"{pruned_avg:.2f} ms", ha='center', va='center', color='white', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'inference_time_comparison.png'))
    plt.close()
    
    # 2. Histogram of speedups for individual images
    speedups = [o/p if p > 0 else 0 for o, p in zip(orig_times, pruned_times)]
    
    plt.figure(figsize=(10, 6))
    plt.hist(speedups, bins=20, color='skyblue', edgecolor='black')
    plt.axvline(x=speedup, color='red', linestyle='--', linewidth=2, label=f'Average: {speedup:.2f}x')
    plt.xlabel('Speedup (Original Time / Pruned Time)')
    plt.ylabel('Number of Images')
    plt.title('Distribution of Speedups Across Images')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'speedup_distribution.png'))
    plt.close()
    
    # 3. Scatter plot of original vs pruned times
    plt.figure(figsize=(10, 6))
    plt.scatter(orig_times, pruned_times, alpha=0.7)
    
    # Add diagonal line (y=x)
    max_time = max(max(orig_times), max(pruned_times))
    plt.plot([0, max_time], [0, max_time], 'r--', label='y=x (No Speedup)')
    
    plt.xlabel('Original Model Inference Time (ms)')
    plt.ylabel('Pruned Model Inference Time (ms)')
    plt.title('Original vs Pruned Inference Times')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'original_vs_pruned_times.png'))
    plt.close()
    
    # 4. Prediction differences (if any)
    different_predictions = sum(1 for orig, pruned in zip(original_results, pruned_results) 
                              if orig['top_indices'][0] != pruned['top_indices'][0])
    
    if different_predictions > 0:
        # Create pie chart of prediction differences
        plt.figure(figsize=(8, 8))
        plt.pie([different_predictions, len(original_results) - different_predictions], 
                labels=['Different', 'Same'], 
                autopct='%1.1f%%', 
                colors=['#ff9999', '#66b3ff'])
        plt.title('Top-1 Prediction Differences Between Models')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'prediction_differences.png'))
        plt.close()
    
    # 5. Top-5 overlap distribution
    top5_overlaps = []
    for orig, pruned in zip(original_results, pruned_results):
        orig_top5 = set(orig['top_indices'][:5])
        pruned_top5 = set(pruned['top_indices'][:5])
        overlap = len(orig_top5.intersection(pruned_top5)) / 5.0
        top5_overlaps.append(overlap * 100)  # Convert to percentage
    
    plt.figure(figsize=(10, 6))
    plt.hist(top5_overlaps, bins=6, color='lightgreen', edgecolor='black')
    plt.axvline(x=np.mean(top5_overlaps), color='red', linestyle='--', linewidth=2, 
                label=f'Average: {np.mean(top5_overlaps):.1f}%')
    plt.xlabel('Top-5 Overlap Percentage')
    plt.ylabel('Number of Images')
    plt.title('Distribution of Top-5 Prediction Overlap')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'top5_overlap_distribution.png'))
    plt.close()
    
    print(f"Comparison plots saved to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Classify images with models')
    parser.add_argument('--original-model', type=str, required=True,
                        help='Path to original model file')
    parser.add_argument('--pruned-model', type=str, required=True,
                        help='Path to pruned model file')
    parser.add_argument('--images', type=str, required=True,
                        help='Directory containing images to classify')
    parser.add_argument('--model-type', type=str, default='resnet18',
                        help='Model architecture type')
    parser.add_argument('--device', type=str, default='cpu',
                        help='Device for inference: cpu or cuda')
    parser.add_argument('--output-dir', type=str, default='comparison_results',
                        help='Directory to save results and plots')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Run classification with original model
    print("\n=== Running Original Model ===")
    original_results = classify_images_with_model(
        args.original_model, 
        args.images, 
        args.model_type, 
        args.device
    )
    
    # Run classification with pruned model
    print("\n=== Running Pruned Model ===")
    pruned_results = classify_images_with_model(
        args.pruned_model, 
        args.images, 
        args.model_type, 
        args.device
    )
    
    # Compare inference times
    if original_results and pruned_results:
        orig_time = sum(r['inference_time'] for r in original_results) / len(original_results)
        pruned_time = sum(r['inference_time'] for r in pruned_results) / len(pruned_results)
        
        speedup = orig_time / pruned_time if pruned_time > 0 else 0
        
        print("\n=== Performance Comparison ===")
        print(f"Original model average inference time: {orig_time*1000:.2f} ms")
        print(f"Pruned model average inference time: {pruned_time*1000:.2f} ms")
        print(f"Speedup: {speedup:.2f}x")
        
        # Compare prediction differences
        different_predictions = 0
        top5_overlap_sum = 0
        
        for orig, pruned in zip(original_results, pruned_results):
            # Check top-1 prediction match
            if orig['top_indices'][0] != pruned['top_indices'][0]:
                different_predictions += 1
            
            # Calculate top-5 overlap
            orig_top5 = set(orig['top_indices'][:5])
            pruned_top5 = set(pruned['top_indices'][:5])
            overlap = len(orig_top5.intersection(pruned_top5)) / 5.0
            top5_overlap_sum += overlap
        
        avg_top5_overlap = top5_overlap_sum / len(original_results)
        
        if different_predictions > 0:
            print(f"\nTop-1 prediction differences: {different_predictions}/{len(original_results)} images ({different_predictions/len(original_results)*100:.2f}%)")
        else:
            print("\nBoth models made identical top-1 predictions for all images.")
            
        print(f"Average top-5 prediction overlap: {avg_top5_overlap*100:.2f}%")
        
        # Save results to CSV
        results = {
            'original_results': original_results,
            'pruned_results': pruned_results
        }
        
        # Extract model names for the output file
        original_model_name = os.path.basename(os.path.dirname(args.original_model))
        pruned_model_name = os.path.basename(os.path.dirname(args.pruned_model))
        
        # If the model names are not descriptive enough, use the full paths
        if not original_model_name or original_model_name == '.':
            original_model_name = os.path.basename(args.original_model).split('.')[0]
        if not pruned_model_name or pruned_model_name == '.':
            pruned_model_name = os.path.basename(args.pruned_model).split('.')[0]
        
        csv_path = os.path.join(args.output_dir, f'{original_model_name}_vs_{pruned_model_name}.csv')
        df = save_results_to_csv(results, csv_path)
        
        # Create comparison plots
        plot_dir = os.path.join(args.output_dir, f'{original_model_name}_vs_{pruned_model_name}_plots')
        create_comparison_plots(results, plot_dir)
