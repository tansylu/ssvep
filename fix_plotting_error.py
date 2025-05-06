import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import os

# Load results
results = []
for layer in range(10, 20):
    sim_file = f"comparison_results/layer_pruning/models_vs_similarity_layer{layer}_10/comparison_summary.csv"
    rand_file = f"comparison_results/layer_pruning/models_vs_random_layer{layer}_10/comparison_summary.csv"
    
    if os.path.exists(sim_file) and os.path.exists(rand_file):
        try:
            sim_df = pd.read_csv(sim_file)
            rand_df = pd.read_csv(rand_file)
            
            # Print column names to debug
            print(f"Layer {layer} similarity columns: {sim_df.columns.tolist()}")
            print(f"Layer {layer} random columns: {rand_df.columns.tolist()}")
            
            # Find match percentage column - look for any column with 'Match' in it
            sim_col = [col for col in sim_df.columns if 'Match' in col]
            rand_col = [col for col in rand_df.columns if 'Match' in col]
            
            if sim_col and rand_col:
                sim_col = sim_col[0]  # Take the first matching column
                rand_col = rand_col[0]
                
                results.append({
                    'Layer': layer,
                    'Similarity': sim_df[sim_col].iloc[0],
                    'Random': rand_df[rand_col].iloc[0],
                    'Difference': sim_df[sim_col].iloc[0] - rand_df[rand_col].iloc[0]
                })
            else:
                print(f"Warning: Could not find Match columns for layer {layer}")
        except Exception as e:
            print(f"Error processing layer {layer}: {e}")

# Create DataFrame
if not results:
    print("No results found. Check if the CSV files exist and have the expected format.")
else:
    comparison_df = pd.DataFrame(results)
    
    # Print DataFrame to debug
    print("\nDataFrame contents:")
    print(comparison_df)
    
    # Plot comparison
    plt.figure(figsize=(12, 6))
    
    # Convert layer to numeric if it's not already
    comparison_df['Layer'] = pd.to_numeric(comparison_df['Layer'])
    
    # Plot using the DataFrame index for x if 'Layer' column is problematic
    sns.lineplot(data=comparison_df, x='Layer', y='Similarity', marker='o', label='Similarity Pruning')
    sns.lineplot(data=comparison_df, x='Layer', y='Random', marker='s', label='Random Pruning')
    plt.title('Similarity vs Random Pruning: Match Percentage by Layer')
    plt.xlabel('Layer')
    plt.ylabel('Match Percentage (%)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig('comparison_plot.png')
    plt.close()
    
    # Plot difference
    plt.figure(figsize=(12, 6))
    bars = plt.bar(comparison_df['Layer'], comparison_df['Difference'], 
                   color=['green' if x > 0 else 'red' for x in comparison_df['Difference']])
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    plt.title('Similarity - Random: Difference in Match Percentage')
    plt.xlabel('Layer')
    plt.ylabel('Difference (%)')
    plt.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., 
                 height + (0.5 if height > 0 else -1.5),
                 f"{height:.1f}%", 
                 ha='center')
    
    plt.savefig('difference_plot.png')
    plt.close()
    
    # Summary statistics
    print("\nSummary Statistics:")
    print(f"Average Match % - Similarity: {comparison_df['Similarity'].mean():.2f}%")
    print(f"Average Match % - Random: {comparison_df['Random'].mean():.2f}%")
    print(f"Average Difference: {comparison_df['Difference'].mean():.2f}%")
    print(f"Layers where Similarity is better: {comparison_df[comparison_df['Difference'] > 0]['Layer'].tolist()}")
    print(f"Layers where Random is better: {comparison_df[comparison_df['Difference'] < 0]['Layer'].tolist()}")