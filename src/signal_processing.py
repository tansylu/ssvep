import numpy as np
from numpy.fft import fft
import os
import csv
from enum import Enum
import db  # Import our database module
from frequency_similarity import calculate_frequency_similarity_score, get_similarity_category

class HarmonicType(Enum):
    """Enum for different types of harmonic checks"""
    ANY = 0       # Check if frequency is a harmonic of any base frequency
    FREQ1 = 1     # Check if frequency is a harmonic of frequency 1
    FREQ2 = 2     # Check if frequency is a harmonic of frequency 2
    INTERMOD = 3  # Check if frequency is an intermodulation product

def is_harmonic_frequency(peak_frequencies, freq1, freq2, harmonic_type=HarmonicType.ANY, tolerance=1, max_harmonic=12, max_intermod_order=5):
    """
    Checks if any of the peak frequencies is a harmonic of the base frequencies.

    Args:
        peak_frequencies: List of frequencies to check
        freq1: First base frequency
        freq2: Second base frequency
        harmonic_type: Type of harmonic check to perform (from HarmonicType enum)
        tolerance: Tolerance for frequency matching
        max_harmonic: Maximum harmonic order to check
        max_intermod_order: Maximum intermodulation order to check

    Returns:
        bool: True if any peak is a harmonic according to the specified type, False otherwise
    """
    # Generate harmonics for frequency 1
    harmonics_freq1 = [n * freq1 for n in range(1, max_harmonic)]

    # Generate harmonics for frequency 2
    harmonics_freq2 = [n * freq2 for n in range(1, max_harmonic)]

    # Combine harmonics for ANY check
    all_harmonics = harmonics_freq1 + harmonics_freq2

    # Generate intermodulation products if needed
    intermod_freqs = []
    if harmonic_type == HarmonicType.INTERMOD:
        for i in range(1, max_intermod_order + 1):
            for j in range(1, max_intermod_order + 1):
                intermod_freqs.append(abs(i * freq1 + j * freq2))
                intermod_freqs.append(abs(i * freq1 - j * freq2))

    # Select the appropriate set of frequencies to check against
    check_freqs = {
        HarmonicType.ANY: all_harmonics,
        HarmonicType.FREQ1: harmonics_freq1,
        HarmonicType.FREQ2: harmonics_freq2,
        HarmonicType.INTERMOD: intermod_freqs
    }[harmonic_type]

    # Check if any peak is a harmonic (ignoring zeros/negatives)
    return any(
        any(abs(peak - h) <= tolerance for h in check_freqs)
        for peak in peak_frequencies if peak > 0
    )

def perform_fourier_transform(activations, reduction_method='mean'):
    """
    Performs FFT on activation time series for each layer and filter.
    Args:
        activations: {layer_id: [frame1_tensor(1,filters,height,width), frame2_tensor...]}
        reduction_method: Method to reduce spatial dimensions ('mean', 'sum', 'max', 'min', 'median', 'std', 'power')
                         'power' calculates mean squared value and may provide better frequency detection
    Returns:
        {layer_id: numpy_array(num_filters, fft_length)}
    """
    reduction_methods = {
        'mean': np.mean,
        'sum': np.sum,
        'max': np.max,
        'min': np.min,
        'median': np.median,
        'std': np.std,
        'power': lambda x: np.mean(x**2)  # Mean squared value (power)
    }

    if reduction_method not in reduction_methods:
        valid_methods = "', '".join(reduction_methods.keys())
        raise ValueError(f"Invalid reduction method: {reduction_method}. Choose from '{valid_methods}'.")  # Lists all available methods

    reduce_fn = reduction_methods[reduction_method]

    fourier_transformed_activations = {}
    for layer_id, frames in activations.items():
        # Skip empty frames
        if not frames or len(frames) == 0 or frames[0] is None:
            print(f"Skipping layer {layer_id} due to empty frames")
            continue

        # Get the shape of the first frame to determine dimensions
        first_frame = frames[0]
        frame_shape = first_frame.shape

        # Check if we have a 4D tensor (batch, filters, height, width)
        if len(frame_shape) == 4:
            num_filters = frame_shape[1]
            num_frames = len(frames)

            # Initialize the Fourier transformed activations array
            fourier_transformed_activations[layer_id] = np.zeros((num_filters, num_frames))

            # Iterate over each filter
            for filter_id in range(num_filters):
                # Extract the temporal sequence for each filter using the specified reduction method
                temporal_sequence = [reduce_fn(frame[0, filter_id, :, :]) for frame in frames]

                # Perform Fourier Transform on the temporal sequence
                fourier_transformed_activations[layer_id][filter_id] = np.abs(fft(temporal_sequence))

                # if fft returns all 0 magnitudes, breakpoint
                if np.all(fourier_transformed_activations[layer_id][filter_id] == 0):
                    print(f"Warning: All zero magnitudes for layer {layer_id}, filter {filter_id}. Check the input data.")
        else:
            print(f"Skipping layer {layer_id} with unexpected shape: {frame_shape}")

    return fourier_transformed_activations

def find_peaks_two_neighbours(signal, threshold_factor=2.0):
    """
    Custom peak detection that only looks at immediate neighbours.
    Args:
        signal: 1D numpy array containing the signal
        threshold_factor: A point is considered a peak if its value is greater than
                         threshold_factor times the average of its neighbours
    Returns:
        numpy array: Indices of detected peaks
    """
    # Signal must be at least 3 points long to find peaks
    if len(signal) < 3:
        return np.array([])

    peaks = []

    # Check each point (except first and last) against neighbours
    for i in range(1, len(signal) - 1):
        # Calculate average of neighbours
        neighbor_avg = (signal[i-1] + signal[i+1]) / 2

        # Check if current point is greater than both neighbours
        # and greater than threshold_factor times the average of neighbours
        if signal[i] > signal[i-1] and signal[i] > signal[i+1] and signal[i] > threshold_factor * neighbor_avg:
            peaks.append(i)

    return np.array(peaks)

def find_peaks_four_neighbours(signal, threshold_factor=1.5):
    """
    Custom peak detection that looks at two neighbours on each side.
    Args:
        signal: 1D numpy array containing the signal
        threshold_factor: A point is considered a peak if its value is greater than threshold_factor times the average of its neighbours
    Returns:
        numpy array: Indices of detected peaks
    """
    # Signal must be at least 5 points long to check 2 neighbours on each side
    if len(signal) < 5:
        return np.array([])

    peaks = []

    # Check each point against two neighbours on each side
    for i in range(2, len(signal) - 2):
        # Calculate average of all four neighbours
        neighbor_avg = (signal[i-2] + signal[i-1] + signal[i+1] + signal[i+2]) / 4

        # Check if current point is greater than ALL four neighbours
        is_local_max = (signal[i] > signal[i-2] and
                        signal[i] > signal[i-1] and
                        signal[i] > signal[i+1] and
                        signal[i] > signal[i+2])

        # And check if it's significantly higher than the average of its neighbours
        is_significant = signal[i] > threshold_factor * neighbor_avg

        if is_local_max and is_significant:
            peaks.append(i)

    return np.array(peaks)

def find_peaks_snr(signal, min_snr=3.0):
    """
    Detects peaks based on local signal-to-noise ratio.
    Args:
        signal: 1D numpy array containing the signal
        min_snr: Minimum SNR to consider a point as a peak
    Returns:
        numpy array: Indices of detected peaks
    """
    if len(signal) < 5:  # Need at least 5 points to estimate noise
        return np.array([])

    peaks = []
    for i in range(2, len(signal) - 2):
        # Calculate local noise level (using points not immediately adjacent)
        local_noise = np.std([signal[i-2], signal[i-1], signal[i+1], signal[i+2]])
        if local_noise == 0:  # Avoid division by zero
            local_noise = 1e-10

        # Calculate SNR
        snr = signal[i] / local_noise

        if (signal[i] > signal[i-1] and signal[i] > signal[i+1] and snr > min_snr):
            peaks.append(i)

    return np.array(peaks)


def find_dominant_frequencies(fourier_transformed_activations, fps, threshold_factor=2.0, num_peaks=3, min_snr=3.0, method='custom'):
    """
    Identifies dominant frequency peaks for each filter in each layer using peak detection.

    Args:
        fourier_transformed_activations (dict): {layer_id: np.array(num_filters, fft_length)}
        fps (float): Sampling rate in Hz (frames per second)
        threshold_factor (float): A point is considered a peak if its value is greater than threshold_factor times the average of its neighbours
    Returns:
        dict: {layer_id: {filter_id: [dominant_frequencies (Hz)]}}
    """
    dominant_frequencies = {}

    for layer_id, layer_fft in fourier_transformed_activations.items():
        num_filters, fft_length = layer_fft.shape
        # Generate frequency bins
        freqs = np.fft.fftfreq(fft_length, d=1/fps)
        freqs = np.fft.fftshift(freqs) # Center zero frequency

        dominant_frequencies[layer_id] = {}

        for filter_id in range(num_filters):
            fft_vals = layer_fft[filter_id]
            magnitudes = np.abs(fft_vals)

            # Get only positive frequencies (excluding DC component)
            positive_mask = freqs > 0
            positive_freqs = freqs[positive_mask]
            positive_magnitudes = magnitudes[positive_mask]

            if method == 'two_neighbours':
                peak_indices = find_peaks_two_neighbours(positive_magnitudes, threshold_factor)
            elif method == 'four_neighbours':
                peak_indices = find_peaks_four_neighbours(positive_magnitudes, threshold_factor)
            elif method == 'snr':
                peak_indices = find_peaks_snr(positive_magnitudes, min_snr)
            else:
                # Default to two neighbours method
                peak_indices = find_peaks_two_neighbours(positive_magnitudes, threshold_factor)

            top_frequencies = []
            if len(peak_indices) > 0:
                # Sort peaks by magnitude (highest first)
                sorted_peaks = sorted(peak_indices, key=lambda idx: positive_magnitudes[idx], reverse=True)

                # Take the top peaks (up to num_peaks)
                for i in range(min(num_peaks, len(sorted_peaks))):
                    peak_idx = sorted_peaks[i]
                    freq = positive_freqs[peak_idx]  # Already positive, no need for abs()
                    top_frequencies.append(freq)
            else:
                # No peaks found, append NaN or zero
                top_frequencies = [np.nan] * num_peaks

            # Pad with zeros if fewer than num_peaks were found
            while len(top_frequencies) < num_peaks:
                top_frequencies.append(np.nan)

            # Store the top frequencies
            dominant_frequencies[layer_id][filter_id] = top_frequencies

    return dominant_frequencies

def save_dominant_frequencies_to_csv(dominant_frequencies, fourier_transformed_activations, output_csv_path, image_path, gif_frequency1, gif_frequency2, fps=30):
    """
    Saves dominant frequencies to CSV file, handling multiple peaks per filter.
    This function appends data to the CSV file, creating a new file if it doesn't exist.
    """
    file_exists = os.path.exists(output_csv_path)
    with open(output_csv_path, mode='a', newline='') as csv_file:
        writer = csv.writer(csv_file)
        # Write an empty row
        writer.writerow([])
        # Then write the header row for the table data if file doesn't exist
        if not file_exists:
            header = ["Image", "Layer ID", "Filter ID"]
            # Add column for each peak
            num_peaks = 3
            for i in range(num_peaks):
                header.append(f"Peak {i+1} Freq")
            header.extend(["GIF Frequency 1", "GIF Frequency 2", "Similarity Score", "Similarity Category"])
            writer.writerow(header)

        # Set harmonic detection parameters
        harmonic_tolerance = 1

        for layer_id in sorted(dominant_frequencies.keys()):
            filters = dominant_frequencies[layer_id]
            for filter_id in sorted(filters.keys()):
                peak_frequencies = filters[filter_id]  # List of top frequencies

                # Get the full FFT data for this filter
                fft_vals = fourier_transformed_activations[layer_id][filter_id]
                magnitudes = np.abs(fft_vals)

                # Generate frequency bins
                fft_length = len(magnitudes)
                freqs = np.fft.fftfreq(fft_length, d=1/fps)

                # Get only positive frequencies
                positive_mask = freqs > 0
                positive_freqs = freqs[positive_mask]
                positive_magnitudes = magnitudes[positive_mask]

                # Calculate similarity score using the full spectrum
                similarity_score, details = calculate_frequency_similarity_score(
                    frequencies=positive_freqs,
                    magnitudes=positive_magnitudes,
                    target_freq1=gif_frequency1,
                    target_freq2=gif_frequency2,
                    tolerance=harmonic_tolerance
                )

                # Get similarity category
                similarity_category = get_similarity_category(similarity_score)

                # Format row data
                row = [image_path, layer_id, filter_id]
                # Add all peak frequencies with formatting
                for peak in peak_frequencies:
                    row.append(f"{peak:.10f}" if peak > 0 else np.nan)
                row.extend([gif_frequency1, gif_frequency2, f"{similarity_score:.4f}", similarity_category])
                writer.writerow(row)

def update_dominant_frequencies_csv(dominant_frequencies, fourier_transformed_activations, output_csv_path, image_path, gif_frequency1, gif_frequency2, fps=30):
    """
    Updates a single CSV file with dominant frequencies across multiple images.
    This function maintains a single CSV file across multiple images, avoiding duplicates.

    Args:
        dominant_frequencies: Dictionary of dominant frequencies {layer_id: {filter_id: [frequencies]}}
        output_csv_path: Path to the CSV file to update
        image_path: Path to the current image being processed
        gif_frequency1: First GIF frequency
        gif_frequency2: Second GIF frequency
    """
    # Check if file exists and read existing data
    existing_data = {}
    file_exists = os.path.exists(output_csv_path)

    if file_exists:
        try:
            with open(output_csv_path, mode='r', newline='') as csv_file:
                reader = csv.reader(csv_file)
                # Skip empty rows and find the header row
                header = None
                for row in reader:
                    if not row:  # Skip empty rows
                        continue
                    if "Image" in row and "Layer ID" in row and "Filter ID" in row:
                        header = row
                        break

                if header:
                    # Get column indices
                    layer_idx = header.index("Layer ID")
                    filter_idx = header.index("Filter ID")

                    # Read existing data
                    for row in reader:
                        if not row:  # Skip empty rows
                            continue

                        # Create a unique key for each layer_id, filter_id combination
                        key = (row[layer_idx], row[filter_idx])

                        # Store the existing data
                        if key not in existing_data:
                            existing_data[key] = []
                        existing_data[key].append(row)
        except Exception as e:
            print(f"Error reading existing CSV file: {e}")
            # If there's an error reading the file, we'll create a new one
            file_exists = False

    # Prepare new data
    new_data = []

    # Set harmonic detection parameters
    harmonic_tolerance = 1

    for layer_id in sorted(dominant_frequencies.keys()):
        filters = dominant_frequencies[layer_id]
        for filter_id in sorted(filters.keys()):
            peak_frequencies = filters[filter_id]  # List of top frequencies

            # Get the full FFT data for this filter
            fft_vals = fourier_transformed_activations[layer_id][filter_id]
            magnitudes = np.abs(fft_vals)

            # Generate frequency bins
            fft_length = len(magnitudes)
            freqs = np.fft.fftfreq(fft_length, d=1/fps)

            # Get only positive frequencies
            positive_mask = freqs > 0
            positive_freqs = freqs[positive_mask]
            positive_magnitudes = magnitudes[positive_mask]

            # Calculate similarity score using the full spectrum
            similarity_score, _ = calculate_frequency_similarity_score(
                frequencies=positive_freqs,
                magnitudes=positive_magnitudes,
                target_freq1=gif_frequency1,
                target_freq2=gif_frequency2,
                tolerance=harmonic_tolerance
            )

            # Get similarity category
            similarity_category = get_similarity_category(similarity_score)

            # Format row data
            row = [image_path, layer_id, filter_id]
            # Add all peak frequencies with formatting
            for peak in peak_frequencies:
                row.append(f"{peak:.10f}" if peak > 0 else np.nan)
            row.extend([gif_frequency1, gif_frequency2, f"{similarity_score:.4f}", similarity_category])

            # Add to new data
            new_data.append(row)

    # Write the updated data to the CSV file
    with open(output_csv_path, mode='w', newline='') as csv_file:
        writer = csv.writer(csv_file)

        # Write header
        header = ["Image", "Layer ID", "Filter ID"]
        # Add column for each peak
        num_peaks = 3
        for i in range(num_peaks):
            header.append(f"Peak {i+1} Freq")
        header.extend(["GIF Frequency 1", "GIF Frequency 2", "Similarity Score", "Similarity Category"])
        writer.writerow(header)

        # Write existing data (if any)
        for key, rows in existing_data.items():
            for row in rows:
                # Check if the existing row has similarity score and category
                # If not, add default values to maintain compatibility
                if len(row) < len(header):
                    # Add default similarity score and category
                    row.extend(["0.0000", "Very Different"])
                writer.writerow(row)

        # Write new data
        for row in new_data:
            writer.writerow(row)

def save_fft_results_to_db(image_path, fourier_transformed_activations, dominant_frequencies, color_format, fps, reduction_method, gif_frequency1=None, gif_frequency2=None):
    """
    Save FFT results to the database.

    Args:
        image_path (str): Path to the image file
        fourier_transformed_activations (dict): Dictionary of FFT results
            {layer_id: numpy_array(num_filters, fft_length)}
        dominant_frequencies (dict): Dictionary of dominant frequencies
            {layer_id: {filter_id: [frequencies]}}
        color_format (str): The color format used (e.g., 'RGB', 'HSV')
        fps (float): Frames per second
        reduction_method (str): Method used to reduce spatial dimensions
        gif_frequency1 (float, optional): First GIF frequency
        gif_frequency2 (float, optional): Second GIF frequency
    """
    # Initialize the database if it doesn't exist
    db.init_db()

    # Get or create the image record
    image_id = db.get_or_create_image(image_path)

    # Create a new run record
    run_id = db.create_run(
        image_id=image_id,
        color_format=color_format,
        fps=fps,
        reduction_method=reduction_method,
        gif_frequency1=gif_frequency1,
        gif_frequency2=gif_frequency2
    )

    # Save the FFT results
    db.save_fft_results(run_id, fourier_transformed_activations)

    # Save the dominant frequencies
    db.save_dominant_frequencies(run_id, dominant_frequencies, gif_frequency1, gif_frequency2)

    return run_id