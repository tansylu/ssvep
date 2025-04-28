import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
from collections import Counter

# Check if CSV file exists
csv_file = 'results/exports/run1_dominant_frequencies.csv'
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

# Analyze similarity scores
similarity_scores = image_df['Similarity Score'].astype(float)

print(f"\nSimilarity Score Statistics:")
print(f"- Mean: {similarity_scores.mean():.4f}")
print(f"- Median: {similarity_scores.median():.4f}")
print(f"- Min: {similarity_scores.min():.4f}")
print(f"- Max: {similarity_scores.max():.4f}")

# Count similarity categories
category_counts = image_df['Similarity Category'].value_counts()
print(f"\nSimilarity Category Distribution:")
for category, count in category_counts.items():
    percent = (count / len(image_df)) * 100
    print(f"- {category}: {count} ({percent:.2f}%)")

# Create a directory for the analysis results
os.makedirs('results/plots/analysis', exist_ok=True)

# Analyze similarity scores by layer
print("\nSimilarity Score by Layer:")
layer_similarity_scores = image_df.groupby('Layer ID')['Similarity Score'].mean().sort_values(ascending=False)
for layer_id, avg_sim in layer_similarity_scores.items():
    layer_count = len(image_df[image_df['Layer ID'] == layer_id])
    print(f"- Layer {layer_id}: {avg_sim:.4f} (from {layer_count} filters)")

# Find the filters with highest similarity scores
print("\nTop 10 filters with highest similarity scores:")
top_similar_filters = image_df.sort_values('Similarity Score', ascending=False).head(10)
for _, row in top_similar_filters.iterrows():
    print(f"- Layer {row['Layer ID']}, Filter {row['Filter ID']}: {row['Similarity Score']:.4f} (Category: {row['Similarity Category']})")

# Plot distribution of similarity scores by layer
plt.figure(figsize=(12, 6))
sns.boxplot(data=image_df, x='Layer ID', y='Similarity Score')
plt.title(f"Distribution of Similarity Scores by Layer for {os.path.basename(current_image)}")
plt.xlabel('Layer ID')
plt.ylabel('Similarity Score')
plt.tight_layout()
plt.savefig('results/plots/analysis/similarity_scores_by_layer.png')
print("\nSaved similarity score distribution plot to 'results/plots/analysis/similarity_scores_by_layer.png'")

# Plot histogram of similarity scores
plt.figure(figsize=(12, 6))
sns.histplot(data=image_df, x='Similarity Score', bins=20)
plt.title(f"Histogram of Similarity Scores for {os.path.basename(current_image)}")
plt.xlabel('Similarity Score')
plt.ylabel('Count')
plt.tight_layout()
plt.savefig('results/plots/analysis/similarity_scores_histogram.png')
print("Saved similarity score histogram to 'results/plots/analysis/similarity_scores_histogram.png'")

# Plot the top filters with highest similarity scores
if len(top_similar_filters) > 0:
    plt.figure(figsize=(12, 6))
    sns.barplot(data=top_similar_filters, x='Filter ID', y='Similarity Score', hue='Layer ID')
    plt.title(f"Top Filters with Highest Similarity Scores for {os.path.basename(current_image)}")
    plt.xlabel('Filter ID')
    plt.ylabel('Similarity Score')
    plt.tight_layout()
    plt.savefig('results/plots/analysis/top_similar_filters.png')
    print("Saved top filters plot to 'results/plots/analysis/top_similar_filters.png'")

# Plot heatmap of similarity scores by layer and filter (for top layers)
top_layers = layer_similarity_scores.head(5).index.tolist()
if top_layers:
    for layer_id in top_layers:
        layer_data = image_df[image_df['Layer ID'] == layer_id].copy()
        if len(layer_data) > 0:
            # Create a pivot table for the heatmap
            max_filter_id = layer_data['Filter ID'].max()
            filter_range = range(int(max_filter_id) + 1)

            # Initialize with NaN values
            heatmap_data = pd.DataFrame(index=[layer_id], columns=filter_range)

            # Fill in the similarity scores
            for _, row in layer_data.iterrows():
                filter_id = int(row['Filter ID'])
                heatmap_data.at[layer_id, filter_id] = row['Similarity Score']

            # Plot the heatmap
            plt.figure(figsize=(20, 2))
            sns.heatmap(heatmap_data, cmap='viridis', vmin=0, vmax=1, cbar_kws={'label': 'Similarity Score'})
            plt.title(f"Similarity Scores for Layer {layer_id}")
            plt.xlabel('Filter ID')
            plt.ylabel('Layer ID')
            plt.tight_layout()
            plt.savefig(f'results/plots/analysis/layer_{layer_id}_heatmap.png')
            print(f"Saved heatmap for Layer {layer_id} to 'results/plots/analysis/layer_{layer_id}_heatmap.png'")

print("\nAnalysis complete!")
