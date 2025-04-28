# SSVEP Analysis Tool Documentation

This directory contains documentation for the SSVEP Analysis Tool.

## Documentation Files

- [Main README](../README.md): Overview of the project, structure, and usage
- [FFT Database README](FFT_DATABASE_README.md): Detailed information on the database functionality
- [Spectrum Generator README](SPECTRUM_GENERATOR_README.md): Guide to generating spectrum visualizations
- [Filter Pruning README](README_filter_pruning.md): Documentation on filter pruning functionality
- [Plot Filters README](README_plot_filters.md): Guide to plotting filter visualizations

## Project Structure

The project is organized into a modular structure:

```
project_root/
├── src/                  # Source code
│   ├── core/             # Core functionality
│   ├── database/         # Database operations
│   ├── analysis/         # Analysis tools
│   ├── utils/            # Utility functions
│   └── main.py           # Main entry point
├── tests/                # Test modules
├── docs/                 # Documentation
├── data/                 # Data files
└── results/              # Results and outputs
```

## Key Features

- **Comprehensive FFT Analysis**: Analyze how each filter in a CNN responds to flickering stimuli
- **Database Integration**: Store and query results across different images, layers, and filters
- **Advanced Visualization**: Generate spectrum plots and visualize frequency patterns
- **Filter Pruning**: Identify and prune filters based on their frequency response
- **Harmonic Analysis**: Detect harmonic and intermodulation patterns in filter responses
- **Batch Processing**: Process multiple images and aggregate statistics

## Command-line Tools

### Main Processing Tool

```
python src/main.py [options]
```

Options:
- `--filter-id FILTER_ID`: Specific filter ID to plot
- `--layer-id LAYER_ID`: Specific layer ID to plot
- `--reduction {mean,sum,max,min,median,std,power}`: Reduction method for spatial dimensions (default: power)
- `--non-intermod`: Only plot spectrums that are not intermodulation products
- `--export-stats OUTPUT_FILE`: Export filter statistics to the specified CSV file
- `--db-only`: Only save results to database, skip CSV files

### Database Query Tool

```
python src/database/query_db.py [command] [options]
```

Commands:
- `list-images`: List all images in the database
- `list-runs`: List all runs in the database
- `filter-stats`: List filter statistics
- `export RUN_ID`: Export a run to CSV
- `plot RUN_ID LAYER_ID FILTER_ID`: Plot FFT data for a specific run, layer, and filter
- `analyze`: Analyze harmonic patterns
- `init`: Initialize the database

### Spectrum Generator

```
python src/analysis/generate_spectrum.py FILTER_ID LAYER_ID [options]
```

### Filter Pruning Tool

```
python src/analysis/prune_filters.py --model MODEL_PATH [options]
```

## Technical Notes

- The tool uses a pre-trained ResNet18 model for feature extraction
- Dominant frequencies are detected using peak finding algorithms with configurable thresholds
- Similarity scores are calculated based on harmonic patterns and energy distribution
- The database schema supports tracking multiple runs, images, and processing parameters
- Filter pruning is based on frequency response similarity to input stimuli
- Multiple peak detection methods are supported (two_neighbours, four_neighbours, snr)
