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
- `--output-dir OUTPUT_DIR`: Base directory to save spectrum plots (default: 'spectrum_plots'). Plots will be saved in subdirectories organized by filter ID and layer ID
- `--csv CSV_FILE`: Export spectrum data to CSV file
- `--max-freq MAX_FREQ`: Maximum frequency to display in Hz (default: 35)
- `--list-only`: Only list runs with data for the specified filter and layer (don't generate plots)
- `--no-zoom`: Disable automatic zooming to the largest magnitude
- `--gif`: Generate an animated GIF of spectrum plots across all runs
- `--gif-duration DURATION`: Duration of each frame in the GIF in seconds (default: 0.5)
- `--max-frames MAX_FRAMES`: Maximum number of frames to include in the GIF

## Examples

### List all runs with data for filter 0, layer 0

```bash
python src/generate_spectrum.py 0 0 --list-only
```

### Generate spectrum plots for filter 0, layer 0 across all runs

```bash
python src/generate_spectrum.py 0 0
```

This will save all plots to the default 'spectrum_plots' directory.

### Generate spectrum plots for filter 0, layer 0 for a specific run

```bash
python src/generate_spectrum.py 0 0 --run-id 1
```

### Save spectrum plots to a custom directory

```bash
python src/generate_spectrum.py 0 0 --output-dir custom_spectrum_plots
```

### Export spectrum data to a CSV file

```bash
python src/generate_spectrum.py 0 0 --csv spectrum_data.csv
```

### Generate plots and export to CSV

```bash
python src/generate_spectrum.py 0 0 --output-dir spectrum_plots --csv spectrum_data.csv
```

### Generate an animated GIF of spectrum plots

```bash
python src/generate_spectrum.py 0 0 --gif
```

### Generate an animated GIF with custom settings

```bash
python src/generate_spectrum.py 0 0 --gif --gif-duration 0.3 --max-frames 20
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

## Output Folder Structure

The script organizes the spectrum plots in a folder structure based on filter ID and layer ID:

```
spectrum_plots/
├── filter_0_layer_0/
│   ├── image1_20250422_144133.png
│   ├── image2_20250422_144134.png
│   └── ...
├── filter_0_layer_1/
│   ├── image1_20250422_144135.png
│   └── ...
└── ...
```

This organization makes it easy to navigate and find specific plots for a particular filter and layer combination.

## Animated GIF Feature

The script can generate an animated GIF that shows how the spectrum changes across different images for a specific filter and layer. This is useful for visualizing patterns and trends in the frequency response.

The GIF is created by:

1. Generating a spectrum plot for each run
2. Using consistent y-axis scaling across all frames (if auto-zoom is enabled)
3. Combining the plots into an animated GIF

The GIF is saved in the same directory structure as the individual plots, with the filename `spectrum_animation_TIMESTAMP.gif`.

You can control the animation speed using the `--gif-duration` parameter, and limit the number of frames using the `--max-frames` parameter.

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
