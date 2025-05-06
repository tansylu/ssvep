import os
import pandas as pd
import argparse
import glob
import re
import matplotlib.pyplot as plt
import numpy as np

def extract_percentage(model_name):
    """Extract percentage from model name if available"""
    match = re.search(r'(\d+)', model_name)
    if match:
        return int(match.group(1))
    return None

def summarize_comparisons(results_dir, output_file):
    """
    Summarize all model comparison results into a single CSV file
    
    Args:
        results_dir: Directory containing comparison results
        output_file: Path to save the summary CSV
    """
    # Find all CSV files in the results directory
    csv_files = glob.glob(os.path.join(results_dir, "*/*.csv"))
    
    if not csv_files:
        print(f"No CSV files found in {results_dir}")
        return
    
    print(f"Found {len(csv_files)} comparison results")
    
    # Collect summary data
    summary_data = []
    
    for csv_file in csv_files:
        model_name = os.path.basename(os.path.dirname(csv_file))
        
        try:
            # Read the CSV file
            df = pd.read_csv(csv_file)
            
            # Extract key metrics
            top1_match_col = [col for col in df.columns if 'Top1_Match' in col]
            top5_overlap_col = [col for col in df.columns if 'Top5_Overlap' in col]
            
            if top1_match_col and top5_overlap_col:
                top1_match = df[top1_match_col[0]].mean() * 100  # Convert to percentage
                top5_overlap = df[top5_overlap_col[0]].mean() * 100  # Convert to percentage
                
                # Calculate average speedup
                speedup = df['Speedup'].mean() if 'Speedup' in df.columns else None
                
                # Extract percentage from model name if available
                percentage = extract_percentage(model_name)
                
                # Determine pruning type
                pruning_type = "Unknown"
                if "similarity" in model_name.lower():
                    pruning_type = "Similarity"
                elif "random" in model_name.lower():
                    pruning_type = "Random"
                elif "layer" in model_name.lower():
                    pruning_type = "Layer"
                
                # Add to summary data
                summary_data.append({
                    'Model': model_name,
                    'Pruning_Type': pruning_type,
                    'Percentage': percentage,
                    'Top1_Match_Pct': top1_match,
                    'Top5_Overlap_Pct': top5_overlap,
                    'Avg_Speedup': speedup
                })
        except Exception as e:
            print(f"Error processing {csv_file}: {e}")
    
    if not summary_data:
        print("No valid data found in CSV files")
        return
    
    # Create summary DataFrame
    summary_df = pd.DataFrame(summary_data)
    
    # Sort by pruning type and percentage
    summary_df = summary_df.sort_values(['Pruning_Type', 'Percentage'])
    
    # Save to CSV
    summary_df.to_csv(output_file, index=False)
    print(f"Summary saved to {output_file}")
    
    # Create plots directory
    plots_dir = os.path.dirname(output_file)
    
    # Create plots
    create_summary_plots(summary_df, plots_dir)

def create_summary_plots(df, output_dir):
    """Create summary plots from the data"""
    # Plot Top1 Match vs Percentage for different pruning types
    plt.figure(figsize=(10, 6))
    
    for pruning_type in df['Pruning_Type'].unique():
        subset = df[df['Pruning_Type'] == pruning_type]
        if not subset.empty and 'Percentage' in subset.columns and not subset['Percentage'].isna().all():
            plt.plot(subset['Percentage'], subset['Top1_Match_Pct'], 'o-', 
                     label=f'{pruning_type} Pruning', linewidth=2, markersize=8)
    
    plt.xlabel('Pruning Percentage (%)')
    plt.ylabel('Top-1 Match Percentage (%)')
    plt.title('Model Accuracy vs Pruning Percentage')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'top1_match_vs_percentage.png'))
    
    # Plot Speedup vs Percentage
    plt.figure(figsize=(10, 6))
    
    for pruning_type in df['Pruning_Type'].unique():
        subset = df[df['Pruning_Type'] == pruning_type]
        if not subset.empty and 'Percentage' in subset.columns and not subset['Percentage'].isna().all():
            plt.plot(subset['Percentage'], subset['Avg_Speedup'], 'o-', 
                     label=f'{pruning_type} Pruning', linewidth=2, markersize=8)
    
    plt.xlabel('Pruning Percentage (%)')
    plt.ylabel('Average Speedup (x)')
    plt.title('Model Speedup vs Pruning Percentage')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'speedup_vs_percentage.png'))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Summarize model comparison results')
    parser.add_argument('--results-dir', type=str, required=True,
                        help='Directory containing comparison results')
    parser.add_argument('--output', type=str, required=True,
                        help='Path to save the summary CSV')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    
    # Summarize comparisons
    summarize_comparisons(args.results_dir, args.output)