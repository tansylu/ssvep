# SSVEP Analysis Tool

## Overview

This tool analyzes neural network activations in response to flickering images and performs Fast Fourier Transform (FFT) analysis to identify dominant frequencies.

## New Database Functionality

The tool now includes a SQLite database for storing FFT results, making it easier to query and analyze the data. See the `FFT_DATABASE_README.md` file for detailed information on using the database.

### Key Features

- Store FFT results in a structured database
- Query and analyze results across different images, layers, and filters
- Export results to CSV files for further analysis
- Visualize FFT spectrums and dominant frequencies

### Command-line Arguments

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

### Database Utilities

Use the `query_db.py` script to query and analyze the database:

```
python src/query_db.py [command] [options]
```

Commands:
- `list-images`: List all images in the database
- `list-runs`: List all runs in the database
- `export RUN_ID`: Export a run to CSV
- `plot RUN_ID LAYER_ID FILTER_ID`: Plot FFT data for a specific run, layer, and filter
- `analyze`: Analyze harmonic patterns
- `init`: Initialize the database

## Notes

- Dominant frequencies are observed to be 5.6 and 11.20 Hz
- The method to find dominant frequencies finds local maximas and takes the biggest local maxima
- Second and third dominant frequencies are now included in the analysis
