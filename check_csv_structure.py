import pandas as pd
import glob
import os

# Check the first CSV file for each type
sim_files = glob.glob("comparison_results/layer_pruning/models_vs_similarity_layer*_10/comparison_summary.csv")
rand_files = glob.glob("comparison_results/layer_pruning/models_vs_random_layer*_10/comparison_summary.csv")

if sim_files:
    print("Sample similarity CSV structure:")
    sample_sim = pd.read_csv(sim_files[0])
    print(sample_sim.head())
    print(f"Columns: {sample_sim.columns.tolist()}")
else:
    print("No similarity CSV files found")

if rand_files:
    print("\nSample random CSV structure:")
    sample_rand = pd.read_csv(rand_files[0])
    print(sample_rand.head())
    print(f"Columns: {sample_rand.columns.tolist()}")
else:
    print("No random CSV files found")

# Check if files exist for all layers
print("\nChecking files for each layer:")
for layer in range(10, 20):
    sim_file = f"comparison_results/layer_pruning/models_vs_similarity_layer{layer}_10/comparison_summary.csv"
    rand_file = f"comparison_results/layer_pruning/models_vs_random_layer{layer}_10/comparison_summary.csv"
    
    sim_exists = os.path.exists(sim_file)
    rand_exists = os.path.exists(rand_file)
    
    print(f"Layer {layer}: Similarity CSV {'exists' if sim_exists else 'missing'}, Random CSV {'exists' if rand_exists else 'missing'}")