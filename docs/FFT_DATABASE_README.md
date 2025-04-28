# FFT Results Database

This document explains how to use the FFT results database system that has been added to the SSVEP project.

## Overview

The FFT results database is a SQLite database that stores the results of Fast Fourier Transform (FFT) analysis performed on neural network activations. The database allows for easy querying and manipulation of the data, making it possible to analyze patterns across different images, layers, and filters.

## Database Schema

The database consists of the following tables:

1. **images**: Stores information about processed images
   - `id`: Primary key
   - `path`: Path to the image file
   - `name`: Name of the image file
   - `created_at`: Timestamp when the record was created

2. **runs**: Stores information about each processing run
   - `id`: Primary key
   - `image_id`: Foreign key to the images table
   - `timestamp`: Timestamp for the run
   - `color_format`: Color format used (e.g., 'RGB', 'HSV')
   - `fps`: Frames per second
   - `gif_frequency1`: First GIF frequency
   - `gif_frequency2`: Second GIF frequency
   - `reduction_method`: Method used to reduce spatial dimensions
   - `created_at`: Timestamp when the record was created

3. **fft_results**: Stores the FFT results for each layer/filter
   - `id`: Primary key
   - `run_id`: Foreign key to the runs table
   - `layer_id`: Layer ID
   - `filter_id`: Filter ID
   - `fft_data`: Serialized numpy array containing FFT data
   - `created_at`: Timestamp when the record was created

4. **dominant_frequencies**: Stores the dominant frequencies for each layer/filter
   - `id`: Primary key
   - `run_id`: Foreign key to the runs table
   - `layer_id`: Layer ID
   - `filter_id`: Filter ID
   - `peak1_freq`: First peak frequency
   - `peak2_freq`: Second peak frequency
   - `peak3_freq`: Third peak frequency
   - `is_harmonic`: Boolean indicating if any peak is a harmonic
   - `harmonic_type`: Type of harmonic (FREQ1, FREQ2, INTERMOD, MULTIPLE)
   - `created_at`: Timestamp when the record was created

## Using the Database

### Saving FFT Results

FFT results are automatically saved to the database when running the main processing script (`main.py`). The script has been modified to call the `save_fft_results_to_db` function after performing FFT analysis.

### Querying the Database

A utility script (`query_db.py`) has been provided to query and analyze the database. The script provides the following commands:

#### List Images

```bash
python src/query_db.py list-images
```

This command lists all images in the database.

#### List Runs

```bash
python src/query_db.py list-runs
```

This command lists all runs in the database. You can filter by image ID:

```bash
python src/query_db.py list-runs --image-id 1
```

#### Export Run to CSV

```bash
python src/query_db.py export 1
```

This command exports the dominant frequencies for run ID 1 to a CSV file. You can specify the output file:

```bash
python src/query_db.py export 1 --output results.csv
```

#### Plot FFT Data

```bash
python src/query_db.py plot 1 0 0
```

This command plots the FFT data for run ID 1, layer 0, filter 0. You can specify the output file:

```bash
python src/query_db.py plot 1 0 0 --output plot.png
```

#### Analyze Harmonics

```bash
python src/query_db.py analyze
```

This command analyzes harmonic patterns across all runs in the database. You can filter by run ID or image ID:

```bash
python src/query_db.py analyze --run-id 1
python src/query_db.py analyze --image-id 1
```

### Programmatic Access

You can also access the database programmatically using the `db` module. Here's an example:

```python
import db

# Initialize the database
db.init_db()

# Get a connection to the database
conn = db.get_connection()

# Query the database
cursor = conn.cursor()
cursor.execute("SELECT * FROM images")
images = cursor.fetchall()

# Close the connection
conn.close()
```

## Database File

The database is stored in a file named `fft_results.db` in the project root directory. This file is created automatically when the database is first used.

## CSV Export

In addition to storing the results in the database, the system still exports dominant frequencies to CSV files for backward compatibility. The CSV files are named `dominant_frequencies_2n.csv`, `dominant_frequencies_4n.csv`, and `dominant_frequencies_snr.csv` for the different peak detection methods.

## Future Enhancements

Possible future enhancements to the database system include:

1. Adding more analysis functions to the `query_db.py` script
2. Adding a web interface for visualizing the data
3. Adding support for exporting the data to other formats (e.g., JSON, Excel)
4. Adding support for comparing results across different runs
