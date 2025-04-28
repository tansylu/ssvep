# Filter Visualization Tool

This script (`plot_best_and_worst_filters.py`) visualizes the best and worst performing filters based on their similarity to expected SSVEP frequencies.

## Important Note on Visualization vs. Ranking

**Key Nuance**: This script ranks filters based on their **average performance across all processed images**, but displays the filter activations from **only one specific run/image**.

This means:
- The "Best" and "Worst" labels reflect a filter's average performance across all images
- The actual FFT spectrum you see in the plot comes from a single specific run (by default, the latest run)
- The similarity score shown in the plot title is recalculated for that specific run and may differ from the average score used for ranking

## Usage

```bash
python plot_best_and_worst_filters.py [options]
```

### Options

- `--stats STATS_FILE`: Path to the filter statistics CSV file (default: filter_stats_20250427_164606.csv)
- `--num NUM_FILTERS`: Number of best/worst filters to plot (default: 20)
- `--output OUTPUT_DIR`: Output directory for plots (default: filter_comparison)
- `--run-id RUN_ID`: Run ID to use for plotting (if not provided, uses the latest run)

## Output

The script creates two directories:
- `filter_comparison/best_filters/`: Contains plots of the best performing filters
- `filter_comparison/worst_filters/`: Contains plots of the worst performing filters

Each directory also contains a summary text file with details about each filter.

## Understanding Similarity Scores

The similarity scores typically range from 0.01 to 0.10, where:
- Higher scores indicate better matching between filter activation patterns and expected frequencies
- Scores represent the proportion of a filter's energy concentrated at the target frequencies
- A score of 0.03 means approximately 3% of the filter's energy is at the expected frequencies

## Interpreting the Plots

Each plot shows:
- The FFT spectrum of the filter's activation pattern for a specific run
- Vertical red lines marking the harmonics of the first target frequency
- Vertical green lines marking the harmonics of the second target frequency
- The title includes the filter's rank based on average performance and its similarity score for the specific run being visualized

## Example

```bash
# Plot the top 30 best and worst filters using data from run ID 42
python plot_best_and_worst_filters.py --stats filter_stats.csv --num 30 --run-id 42
```

## Limitations

- The ranking is based on average performance, which may not reflect a filter's performance on any specific image
- The visualization shows only one run, which may not be representative of the filter's typical behavior
- Similarity scores are relatively low (0.01-0.10 range) due to normalization against the total energy across all frequencies
