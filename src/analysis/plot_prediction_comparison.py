import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import glob

# Directory containing comparison results
base_dir = '/Users/poyrazguler/Desktop/bitirme/ssvep-main/comparison_results'

# Patterns for all types of pruning
similarity_pattern = 'models_vs_similarity_*'
random_pattern = 'models_vs_random_*'
highest_pattern = 'models_vs_highest_*'  # New pattern for highest similarity pruning

def extract_data_from_directories(pattern):
    """Extract pruning percentages and Top1 match percentages from directories matching pattern"""
    result_dirs = glob.glob(os.path.join(base_dir, pattern))
    
    data = []  # List to store (percentage, match_pct) tuples
    
    for result_dir in result_dirs:
        # Extract pruning percentage from directory name
        dir_name = os.path.basename(result_dir)
        try:
            percentage = int(dir_name.split('_')[-1])
            
            # Look for summary CSV files
            summary_files = glob.glob(os.path.join(result_dir, '*_summary.csv'))
            if summary_files:
                df = pd.read_csv(summary_files[0])
                
                # Extract Top1 match percentage
                if 'Top1_Match_Percentage' in df.columns:
                    match_pct = df['Top1_Match_Percentage'].iloc[0]
                    data.append((percentage, match_pct))
        except Exception as e:
            print(f"Error processing {dir_name}: {e}")
    
    # Sort by percentage
    data.sort(key=lambda x: x[0])
    
    # Unzip the sorted data
    pruning_percentages, top1_match_percentages = zip(*data) if data else ([], [])
    
    return list(pruning_percentages), list(top1_match_percentages)

# Extract data for all pruning methods
similarity_percentages, similarity_matches = extract_data_from_directories(similarity_pattern)
random_percentages, random_matches = extract_data_from_directories(random_pattern)
highest_percentages, highest_matches = extract_data_from_directories(highest_pattern)  # New data for highest similarity

# Create the line plot
plt.figure(figsize=(12, 7))

# Plot lines for all methods
if similarity_percentages and similarity_matches:
    plt.plot(similarity_percentages, similarity_matches, 'o-', color='blue', linewidth=2, 
             label='Similarity-based Pruning (Lowest)', markersize=8)

if random_percentages and random_matches:
    plt.plot(random_percentages, random_matches, 's-', color='red', linewidth=2, 
             label='Random Pruning', markersize=8)

if highest_percentages and highest_matches:
    plt.plot(highest_percentages, highest_matches, '^-', color='green', linewidth=2, 
             label='Similarity-based Pruning (Highest)', markersize=8)

# Add labels and title
plt.xlabel('Pruning Percentage (%)', fontsize=12)
plt.ylabel('Prediction Match with Original ResNet (%)', fontsize=12)
plt.title('Comparison of Pruning Methods: Prediction Agreement with Original ResNet-18', fontsize=14)
plt.grid(True, alpha=0.3)
plt.legend(fontsize=12)

# Set y-axis to start from 0 for better visual comparison
plt.ylim(0, 105)

# Add horizontal line at 100% for reference
plt.axhline(y=100, color='green', linestyle='--', alpha=0.5)

# Annotate data points with values
for x, y in zip(similarity_percentages, similarity_matches):
    plt.annotate(f"{y:.1f}%", (x, y), textcoords="offset points", 
                 xytext=(0,10), ha='center', fontsize=9)

for x, y in zip(random_percentages, random_matches):
    plt.annotate(f"{y:.1f}%", (x, y), textcoords="offset points", 
                 xytext=(0,-15), ha='center', fontsize=9)

for x, y in zip(highest_percentages, highest_matches):
    plt.annotate(f"{y:.1f}%", (x, y), textcoords="offset points", 
                 xytext=(0,10), ha='center', fontsize=9)

# Set x-ticks to match the actual percentages
all_percentages = sorted(set(similarity_percentages + random_percentages + highest_percentages))
plt.xticks(all_percentages)

plt.tight_layout()

# Save the plot
output_path = os.path.join(base_dir, 'pruning_methods_comparison.png')
plt.savefig(output_path)
print(f"Plot saved to {output_path}")
plt.show()

# Also save the data as CSV for reference
# Combine the data for all methods
combined_data = []
for p, m in zip(similarity_percentages, similarity_matches):
    combined_data.append((p, m, 'Similarity (Lowest)'))
for p, m in zip(random_percentages, random_matches):
    combined_data.append((p, m, 'Random'))
for p, m in zip(highest_percentages, highest_matches):
    combined_data.append((p, m, 'Similarity (Highest)'))

# Sort by percentage
combined_data.sort(key=lambda x: x[0])

# Create DataFrame
comparison_data = pd.DataFrame(combined_data, columns=['Pruning_Percentage', 'Prediction_Match_Percentage', 'Pruning_Method'])

csv_path = os.path.join(base_dir, 'pruning_methods_comparison.csv')
comparison_data.to_csv(csv_path, index=False)
print(f"Comparison data saved to {csv_path}")
