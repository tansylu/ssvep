# Filter Pruning for SSVEP Models

This tool allows you to prune (remove) filters from your neural network model based on their SSVEP frequency response. By removing filters that don't respond well to the target frequencies, you can potentially:

1. Reduce model size
2. Improve inference speed
3. Potentially improve performance by removing noisy or irrelevant filters

## How It Works

The pruning algorithm:

1. Uses the similarity scores calculated during SSVEP analysis
2. Identifies filters with the lowest scores (worst performers)
3. Extracts the model weights and sets the weights of poor-performing filters to zero
4. Saves the pruned weights and visualization of the pruning process

## Usage

```bash
python prune_filters.py --model path/to/original/model.pth --stats filter_stats.csv
```

### Options

- `--model MODEL`: Path to the original model file (required)
- `--stats STATS`: Path to the filter statistics CSV file (default: filter_stats.csv)
- `--output OUTPUT`: Output directory for the pruned model (default: pruned_model)
- `--percentage PERCENTAGE`: Percentage of worst filters to prune (0-1, default: 0.3)
- `--min-score MIN_SCORE`: Minimum similarity score threshold (filters below this will be pruned)
- `--layer-mapping MAPPING`: Path to a CSV file mapping layer IDs to layer names

You can prune filters either by percentage (e.g., the worst 30%) or by setting a minimum score threshold.

## Output

The script produces:

1. A pruned weights file (`pruned_weights.npz`)
2. A visualization of which filters were pruned (`pruning_visualization.png`)
3. A text file listing all pruned filters and their scores (`pruned_filters.txt`)

## Example

```bash
# Prune the worst 20% of filters
python prune_filters.py --model models/vgg16.h5 --stats filter_stats.csv --percentage 0.2

# Prune all filters with similarity score below 0.02
python prune_filters.py --model models/vgg16.h5 --stats filter_stats.csv --min-score 0.02
```

## Important Notes

1. **TensorFlow-Free Implementation**: This script works without requiring TensorFlow, making it compatible with any environment.

2. **Pruning Method**: This implementation "prunes" filters by setting their weights to zero, rather than actually removing them from the model architecture. This preserves the model structure while disabling the pruned filters.

3. **Layer Naming**: The script attempts to match layer IDs to layer names in the model file. If you have a non-standard naming convention, you may need to provide a layer mapping file.

4. **Using the Pruned Weights**: To use the pruned weights, you'll need to load the NPZ file in your model code and apply the weights to your model.

5. **Evaluation**: After pruning, you should evaluate the model with the pruned weights to ensure it still performs adequately for your task.

## How to Choose What to Prune

When deciding how many filters to prune, consider:

1. **Score Distribution**: Look at the distribution of similarity scores to identify natural thresholds
2. **Model Performance**: Start with a small percentage (10-20%) and gradually increase while monitoring performance
3. **Layer Importance**: Some layers may be more critical than others; consider pruning more aggressively from less important layers

The visualization produced by this script can help you make these decisions by showing the distribution of scores and which filters were pruned.

## Loading Pruned Weights

To use the pruned weights in your model:

```python
import numpy as np
from tensorflow.keras.models import load_model

# Load your original model
model = load_model('original_model.h5')

# Load the pruned weights
pruned_weights = np.load('pruned_weights.npz')

# Apply the pruned weights to your model
# This is a simplified example and may need adjustment for your specific model
for i, layer in enumerate(model.layers):
    if 'conv' in layer.name.lower():
        # Find the corresponding weights in the pruned weights
        weights_key = None
        biases_key = None

        for key in pruned_weights.keys():
            if layer.name in key and 'kernel' in key:
                weights_key = key
            elif layer.name in key and 'bias' in key:
                biases_key = key

        if weights_key is not None and biases_key is not None:
            # Set the layer weights
            layer.set_weights([pruned_weights[weights_key], pruned_weights[biases_key]])

# Save the model with pruned weights
model.save('model_with_pruned_weights.h5')
```
