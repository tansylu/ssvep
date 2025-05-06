import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load the data
df = pd.read_csv('/Users/tansylu/Documents/ssvep/filter_stats (2).csv')

# Create a figure
plt.figure(figsize=(10, 6))

# Create histogram of similarity scores
sns.histplot(df['Avg Similarity Score'], bins=30, kde=True)

# Add vertical line at mean
mean_score = df['Avg Similarity Score'].mean()
plt.axvline(x=mean_score, color='r', linestyle='--', 
            label=f'Mean: {mean_score:.4f}')

# Add vertical line at median
median_score = df['Avg Similarity Score'].median()
plt.axvline(x=median_score, color='g', linestyle=':', 
            label=f'Median: {median_score:.4f}')

# Add title and labels
plt.title('Distribution of Average Similarity Scores', fontsize=14)
plt.xlabel('Average Similarity Score', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.legend()

# Improve appearance
plt.tight_layout()

# Save the plot
plt.savefig('similarity_scores_distribution.png', dpi=300)

# Show the plot
plt.show()

# Also create a boxplot by layer to see distribution across layers
plt.figure(figsize=(12, 6))
sns.boxplot(x='Layer', y='Avg Similarity Score', data=df)
plt.title('Distribution of Similarity Scores by Layer', fontsize=14)
plt.xlabel('Layer', fontsize=12)
plt.ylabel('Average Similarity Score', fontsize=12)
plt.tight_layout()
plt.savefig('similarity_scores_by_layer.png', dpi=300)
plt.show()