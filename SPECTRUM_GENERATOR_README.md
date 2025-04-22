# Spectrum Generator

This tool allows you to generate spectrum data from the FFT database for specific filter IDs and layers.

## Overview

The `generate_spectrum.py` script provides functionality to:

1. List all runs that have data for a specific filter ID and layer
2. Generate spectrum plots for a specific filter ID and layer across all runs or for a specific run
3. Export spectrum data to a CSV file for further analysis

## Usage

```bash
python src/generate_spectrum.py FILTER_ID LAYER_ID [options]
```

### Required Arguments

- `FILTER_ID`: The filter ID to generate spectrum for
- `LAYER_ID`: The layer ID to generate spectrum for

### Optional Arguments

- `--run-id RUN_ID`: Specific run ID to use (if you want to analyze just one run)
- `--output-dir OUTPUT_DIR`: Directory to save spectrum plots (if not provided, plots will be displayed)
- `--csv CSV_FILE`: Export spectrum data to CSV file
- `--max-freq MAX_FREQ`: Maximum frequency to display in Hz (default: 35)
- `--list-only`: Only list runs with data for the specified filter and layer (don't generate plots)
- `--no-zoom`: Disable automatic zooming to the largest magnitude

## Examples

### List all runs with data for filter 0, layer 0

```bash
python src/generate_spectrum.py 0 0 --list-only
```

### Generate spectrum plots for filter 0, layer 0 across all runs

```bash
python src/generate_spectrum.py 0 0
```

### Generate spectrum plots for filter 0, layer 0 for a specific run

```bash
python src/generate_spectrum.py 0 0 --run-id 1
```

### Save spectrum plots to a directory

```bash
python src/generate_spectrum.py 0 0 --output-dir spectrum_plots
```

### Export spectrum data to a CSV file

```bash
python src/generate_spectrum.py 0 0 --csv spectrum_data.csv
```

### Generate plots and export to CSV

```bash
python src/generate_spectrum.py 0 0 --output-dir spectrum_plots --csv spectrum_data.csv
```

## CSV Output Format

The CSV file contains the following columns:

- `run_id`: The run ID
- `image_name`: The name of the image
- `layer_id`: The layer ID
- `filter_id`: The filter ID
- `frequency`: The frequency in Hz
- `magnitude`: The magnitude of the frequency component
- `gif_frequency1`: The first GIF frequency (if available)
- `gif_frequency2`: The second GIF frequency (if available)
- `timestamp`: The timestamp of the run

This format allows for easy analysis of the spectrum data using tools like Excel, Python, or R.

## Auto-Zoom Feature

By default, the script automatically zooms in on the spectrum by:

1. Setting the y-axis limit to the maximum magnitude present in the data (plus a small padding)
2. Annotating the frequency with the maximum magnitude

This makes it easier to identify the dominant frequencies in the spectrum, especially when there are large differences in magnitude between different frequency components.

You can disable this feature using the `--no-zoom` flag if you prefer to see the full range of the y-axis.

## Notes

- The script requires the FFT database to be initialized. If you haven't run the main processing script yet, you'll need to do that first.
- The spectrum plots include vertical lines for dominant frequencies and GIF frequencies to help with analysis.
- The CSV export includes all frequency components, not just the dominant ones, allowing for more detailed analysis.
- The auto-zoom feature helps to focus on the most significant frequency components in the spectrum.
