import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
from collections import Counter

# Check if CSV file exists
csv_file = 'run1_dominant_frequencies.csv'
if not os.path.exists(csv_file):
    print(f"Error: CSV file '{csv_file}' not found.")
    sys.exit(1)

# Load the CSV data
print(f"Loading data from {csv_file}...")
df = pd.read_csv(csv_file)
df = df.dropna(how='all')

# Get the current image name (assuming it's the first one in the CSV)
current_image = df['Image'].iloc[0]
print(f"\nAnalyzing data for image: {current_image}")

# Filter data for the current image
image_df = df[df['Image'] == current_image]
print(f"Found {len(image_df)} entries for this image")

# Count "Different" vs "Same" flags
flag_counts = image_df['Flag'].value_counts()
different_count = flag_counts.get('Different', 0)
same_count = flag_counts.get('Same', 0)
total_count = different_count + same_count
different_percent = (different_count / total_count) * 100 if total_count > 0 else 0

print(f"\nFlag distribution:")
print(f"- Different: {different_count} ({different_percent:.2f}%)")
print(f"- Same: {same_count} ({100 - different_percent:.2f}%)")

# Analyze "Different" flags by layer
different_df = image_df[image_df['Flag'] == 'Different']
if len(different_df) > 0:
    print("\nDistribution of 'Different' flags by layer:")
    layer_counts = different_df['Layer ID'].value_counts().sort_index()
    for layer_id, count in layer_counts.items():
        layer_percent = (count / len(different_df)) * 100
        print(f"- Layer {layer_id}: {count} ({layer_percent:.2f}%)")
    
    # Find the worst filters (most likely to be flagged as "Different")
    print("\nTop 10 filters with 'Different' flags:")
    filter_counts = different_df.groupby(['Layer ID', 'Filter ID']).size().reset_index(name='count')
    filter_counts = filter_counts.sort_values('count', ascending=False).head(10)
    for _, row in filter_counts.iterrows():
        print(f"- Layer {row['Layer ID']}, Filter {row['Filter ID']}: {row['count']} occurrences")
    
    # Create a directory for the analysis results
    os.makedirs('analysis_results', exist_ok=True)
    
    # Plot distribution of "Different" flags by layer
    plt.figure(figsize=(12, 6))
    sns.countplot(data=different_df, x='Layer ID', order=sorted(different_df['Layer ID'].unique()))
    plt.title(f"Distribution of 'Different' Flags by Layer for {os.path.basename(current_image)}")
    plt.xlabel('Layer ID')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig('analysis_results/different_flags_by_layer.png')
    print("\nSaved layer distribution plot to 'analysis_results/different_flags_by_layer.png'")
    
    # Plot the top filters with "Different" flags
    if len(filter_counts) > 0:
        plt.figure(figsize=(12, 6))
        sns.barplot(data=filter_counts, x='Filter ID', y='count', hue='Layer ID')
        plt.title(f"Top Filters with 'Different' Flags for {os.path.basename(current_image)}")
        plt.xlabel('Filter ID')
        plt.ylabel('Count')
        plt.tight_layout()
        plt.savefig('analysis_results/top_different_filters.png')
        print("Saved top filters plot to 'analysis_results/top_different_filters.png'")
else:
    print("\nNo 'Different' flags found for this image.")

print("\nAnalysis complete!")
