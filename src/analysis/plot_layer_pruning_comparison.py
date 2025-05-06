import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import glob
import seaborn as sns

# Directory containing comparison results
base_dir = 'comparison_results/layer_pruning/'

def extract_data_from_directories():
    """Extract layer pruning data from all result directories"""
    result_dirs = glob.glob(os.path.join(base_dir, 'models_vs_*'))
    data = []  # List to store (layer, percentage, pruning_type, match_pct) tuples
    
    for result_dir in result_dirs:
        # Extract information from directory name
        dir_name = os.path.basename(result_dir)
        
        if 'random' in dir_name:
            pruning_type = 'Random'
            parts = dir_name.split('_')
            layer_part = [p for p in parts if 'layer' in p][0]
            layer_id = int(layer_part.replace('layer', ''))
            percentage_part = parts[-1]
            percentage = int(percentage_part)
        else :
            pruning_type = 'Similarity'
            parts = dir_name.split('_')
            layer_part = [p for p in parts if 'layer' in p][0]
            layer_id = int(layer_part.replace('layer', ''))
            percentage_part = parts[-1]
            percentage = int(percentage_part)
        
        # Find CSV file in the directory
        csv_files = glob.glob(os.path.join(result_dir, '*.csv'))
        print(f"Found {len(csv_files)} CSV files in {result_dir}")
        for csv_file in csv_files:
            print(f"Found {csv_file} CSV files in {result_dir}")
            if "2_summary" in os.path.basename(csv_file):
                print(f"Processing CSV file: {csv_file}")
                try:
                    df = pd.read_csv(csv_file)
                    # Extract match percentage from summary data
                    top1_match = df.get('Top1_Match_Percentage', 0)
                    print(f"Match percentage: {top1_match}")
                    if isinstance(top1_match, pd.Series):
                        top1_match = top1_match.iloc[0]
                    
                    # Add to data list
                    data.append({
                        'Layer': layer_id,
                        'Percentage': percentage,
                        'Pruning_Type': pruning_type,
                        'Match_Percentage': top1_match
                    })
                except Exception as e:
                    print(f"Error processing {csv_file}: {e}")
    
    # Create DataFrame from collected data
    if data:
        results_df = pd.DataFrame(data)
        # Save to CSV
        output_path = os.path.join(base_dir, 'layer_pruning_comparison.csv')
        results_df.to_csv(output_path, index=False)
        print(f"Layer pruning comparison data saved to {output_path}")
        return results_df
    else:
        print("No data found")
        return None

if __name__ == "__main__":
    extract_data_from_directories()
