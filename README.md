# SSVEP Analysis Tool

## Overview

This tool analyzes neural network activations in response to flickering images and performs Fast Fourier Transform (FFT) analysis to identify dominant frequencies. It's designed to study how convolutional neural networks respond to Steady-State Visual Evoked Potentials (SSVEP) stimuli.

## Project Structure

The project is organized into a modular structure for better maintainability and extensibility:

```
project_root/
├── src/                  # Source code
│   ├── core/             # Core functionality
│   │   ├── model.py      # Neural network model functionality
│   │   └── signal_processing.py # Signal processing and FFT analysis
│   ├── database/         # Database operations
│   │   ├── db.py         # Core database operations
│   │   ├── db_stats.py   # Database statistics
│   │   └── query_db.py   # Database querying utilities
│   ├── analysis/         # Analysis tools
│   │   ├── frequency_similarity.py     # Frequency similarity calculations
│   │   ├── analyze_similarity_scores.py # Analysis of similarity scores
│   │   ├── generate_spectrum.py        # Spectrum generation
│   │   ├── plot_best_and_worst_filters.py # Filter visualization
│   │   └── prune_filters.py           # Filter pruning utilities
│   ├── utils/            # Utility functions
│   │   └── flicker_image.py # Image flickering utilities
│   └── main.py           # Main entry point
├── tests/                # Test modules
├── docs/                 # Documentation
├── data/                 # Data files
│   ├── raw/              # Original images
│   ├── processed/        # Processed frames and activations
│   ├── models/           # Model weights
│   └── fft_results.db    # SQLite database for FFT results
└── results/              # Results and outputs
    ├── spectrums/        # Spectrum plots
    ├── plots/            # Other plots
    └── exports/          # Exported data (CSV files)
```

## Core Logic and Workflow

The tool follows this general workflow:

1. **Image Flickering**:
   - Takes input images and creates flickering versions with specific frequencies
   - Supports different color formats (RGB, HSV, LAB)
   - Can apply different frequencies to left and right halves of the image

2. **Neural Network Processing**:
   - Passes flickering images through a pre-trained ResNet18 model
   - Extracts activations from each layer and filter
   - Supports various spatial reduction methods (mean, sum, max, min, median, std, power)

3. **Frequency Analysis**:
   - Performs Fast Fourier Transform (FFT) on the temporal activation patterns
   - Identifies dominant frequencies in each filter's response
   - Calculates similarity scores between filter responses and input frequencies

4. **Database Storage**:
   - Stores all FFT results in a SQLite database
   - Organizes data by image, run, layer, and filter
   - Tracks similarity scores and frequency patterns

5. **Analysis and Visualization**:
   - Generates spectrum plots for each filter
   - Identifies filters with high/low similarity to input frequencies
   - Supports filter pruning based on frequency response
   - Exports statistics and results to CSV files

## Key Features

- **Comprehensive FFT Analysis**: Analyze how each filter in a CNN responds to flickering stimuli
- **Database Integration**: Store and query results across different images, layers, and filters
- **Advanced Visualization**: Generate spectrum plots and visualize frequency patterns
- **Filter Pruning**: Identify and prune filters based on their frequency response
- **Harmonic Analysis**: Detect harmonic and intermodulation patterns in filter responses
- **Batch Processing**: Process multiple images and aggregate statistics

## Technical Notes

- The tool uses a pre-trained ResNet18 model for feature extraction
- Dominant frequencies are detected using peak finding algorithms with configurable thresholds
- Similarity scores are calculated based on harmonic patterns and energy distribution
- The database schema supports tracking multiple runs, images, and processing parameters
- Filter pruning is based on frequency response similarity to input stimuli
- Multiple peak detection methods are supported (two_neighbours, four_neighbours, snr)
