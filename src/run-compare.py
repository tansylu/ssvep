from datetime import datetime
import os
import random
import torch
import torchvision.transforms as transforms
from flicker_image import flicker_image_hh_and_save_gif
from model import get_activations, load_activations, save_activations, init_model
from signal_processing import perform_fourier_transform, find_dominant_frequencies, save_dominant_frequencies_to_csv, is_harmonic_frequency, HarmonicType
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import csv
import sys
import json
import hashlib


# Create a directory for comparison results
COMPARISON_DIR = "comparison_results"
os.makedirs(COMPARISON_DIR, exist_ok=True)

def log_step(step_name, data=None, run_id="run1"):
    """Log a step in the pipeline with optional data for comparison"""
    log_file = os.path.join(COMPARISON_DIR, f"{run_id}_log.txt")
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")

    with open(log_file, "a") as f:
        f.write(f"[{timestamp}] {step_name}\n")
        if data is not None:
            if isinstance(data, np.ndarray):
                f.write(f"  Shape: {data.shape}, Sum: {np.sum(data)}, Mean: {np.mean(data)}, Hash: {hash_array(data)}\n")
            elif isinstance(data, dict):
                f.write(f"  Dict with {len(data)} keys\n")
                for key in sorted(data.keys()):
                    if isinstance(data[key], np.ndarray):
                        f.write(f"    {key}: Shape {data[key].shape}, Sum: {np.sum(data[key])}, Hash: {hash_array(data[key])}\n")
                    elif isinstance(data[key], list):
                        f.write(f"    {key}: List with {len(data[key])} items\n")
                    else:
                        f.write(f"    {key}: {data[key]}\n")
            else:
                f.write(f"  {data}\n")
        f.write("\n")

def hash_array(arr):
    """Create a hash of a numpy array for comparison"""
    return hashlib.md5(arr.tobytes()).hexdigest()

def save_frames(frames, frames_dir):
    os.makedirs(frames_dir, exist_ok=True)
    for i, frame in enumerate(frames):
        frame_image = Image.fromarray(frame.astype(np.uint8))  # Convert numpy array to PIL Image
        frame_path = os.path.join(frames_dir, f"frame_{i}.png")
        frame_image.save(frame_path)
    print(f"Frames saved in '{frames_dir}' directory.")

def load_frames(frames_dir):
    frames = []
    for frame_file in sorted(os.listdir(frames_dir)):
        frame_path = os.path.join(frames_dir, frame_file)
        frame_image = Image.open(frame_path)
        frames.append(np.array(frame_image))
    return frames

def perform_activations(model, frames, preprocess_seqn, run_id="run1"):
    log_step("Starting activation extraction", run_id=run_id)

    # Process each frame with explicit CPU usage
    activations = {}
    hooks = []
    layer_idx_map = {}

    # Define hook function
    def hook_fn(layer_idx):
        def _hook(_module, _input, output):
            # Store the output of the layer
            if layer_idx not in activations:
                activations[layer_idx] = []
            # Convert to numpy and store
            activations[layer_idx].append(output.detach().cpu().numpy())
        return _hook

    # Register hooks for layers we're interested in
    idx = 0
    for name, module in model.named_modules():
        # Only include Conv2d layers (exclude Linear/FC layers)
        if isinstance(module, torch.nn.Conv2d):
            # Skip downsample layers
            if 'downsample' not in name:
                layer_idx_map[name] = idx
                hooks.append(module.register_forward_hook(hook_fn(idx)))
            idx += 1

    # Process each frame
    for frame_idx, frame in enumerate(frames):
        img = Image.fromarray(frame)
        x = preprocess_seqn(img).unsqueeze(0).cpu()  # Add batch dimension and ensure CPU
        with torch.no_grad():  # disable gradient computation
            _ = model(x)  # Forward pass through the model

        if frame_idx % 100 == 0 and frame_idx > 0:
            print(f"Processed {frame_idx}/{len(frames)} frames")

    # Remove hooks
    for hook in hooks:
        hook.remove()

    log_step("Completed activation extraction", activations, run_id=run_id)
    return activations

def plot_and_save_spectrums(fourier_transformed_activations, output_dir, fps, dominant_frequencies, gif_frequency1, gif_frequency2, run_id="run1"):
    """
    Plots and saves the spectrums of the Fourier Transformed activations.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for layer_id, layer_fft in fourier_transformed_activations.items():
        num_filters, fft_length = layer_fft.shape
        freqs = np.fft.fftfreq(fft_length, d=1/fps)  # freq in Hz
        freqs = np.fft.fftshift(freqs)  # Shift the zero frequency to the center

        for filter_id in range(num_filters):
            peak_frequencies = dominant_frequencies[layer_id][filter_id]
            # Check if any of the dominant frequencies is a harmonic of the GIF frequency
            harmonic_tolerance = 0.1

            # Use the global method to check for harmonics
            is_harmonic = is_harmonic_frequency(
                peak_frequencies=peak_frequencies,
                freq1=gif_frequency1,
                freq2=gif_frequency2,
                harmonic_type=HarmonicType.ANY,
                tolerance=harmonic_tolerance
            )

            plt.figure(figsize=(10, 5))
            # Exclude the DC component by starting from index 1
            plt.bar(freqs[1:], np.abs(layer_fft[filter_id][1:]), width=0.05, label=f'Filter {filter_id}')
            plt.title(f'Layer {layer_id+1} Filter {filter_id} Spectrum')
            plt.xlabel('Frequency')
            plt.ylabel('Magnitude')
            plt.legend()
            # Add ticks at the target frequency and its harmonics
            harmonic_ticks1 = [n * gif_frequency1 for n in range(-2, 3)]
            harmonic_ticks2 = [n * gif_frequency2 for n in range(-2, 3)]
            for tick in harmonic_ticks1:
                plt.axvline(x=tick, color='r', linestyle='--', linewidth=0.5, label='f1 harmonic' if tick == gif_frequency1 else "")
            for tick in harmonic_ticks2:
                plt.axvline(x=tick, color='g', linestyle='--', linewidth=0.5, label='f2 harmonic' if tick == gif_frequency2 else "")

            plot_path = os.path.join(output_dir, f'layer_{layer_id}_filter_{filter_id}_spectrum.png')
            plt.savefig(plot_path)
            plt.close()
            print(f'Saved spectrum plot for Layer {layer_id+1} Filter {filter_id} at {plot_path}')

def run_pipeline(image_path, run_id="run1", clean_run=False):
    """Run the full pipeline on a single image and log each step"""
    log_step("Starting pipeline", f"Image: {image_path}, Run ID: {run_id}", run_id=run_id)

    # Define preprocessing transformations
    print("Creating preprocessing sequence...")
    preprocess_seqn = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    log_step("Created preprocessing sequence", run_id=run_id)

    # Extract base name without extension to use in output names
    base_name, _ = os.path.splitext(os.path.basename(image_path))
    print(f"\nProcessing image: {image_path}")

    # Set up paths for this run
    color_format = "RGB"
    gif_path_modified = f"{run_id}_{base_name}_{color_format.lower()}.gif"
    frames_dir = f"{run_id}_frames_{base_name}_{color_format.lower()}"
    activations_output_dir = f'{run_id}_activations_output_{base_name}_{color_format.lower()}'

    # Clean previous run files if requested
    if clean_run:
        for dir_path in [frames_dir, activations_output_dir]:
            if os.path.exists(dir_path):
                print(f"Removing previous run directory: {dir_path}")
                import shutil
                shutil.rmtree(dir_path)

    # Generate frames
    if not os.path.exists(gif_path_modified) or clean_run:
        log_step("Generating flicker frames", run_id=run_id)
        frames = flicker_image_hh_and_save_gif(
            image_path=image_path,
            output_gif=gif_path_modified,
            duration=5,
            frequency1=5,
            frequency2=6,
            fps=24,
            color_format=color_format
        )
        log_step("Generated flicker frames", f"Number of frames: {len(frames)}", run_id=run_id)

        # Save frames as images
        save_frames(frames, frames_dir)
        log_step("Saved frames to directory", frames_dir, run_id=run_id)
        print(f"GIF saved as '{gif_path_modified}'.")
    else:
        print(f"GIF '{gif_path_modified}' already exists. Loading frames from '{frames_dir}'...")
        frames = load_frames(frames_dir)
        log_step("Loaded existing frames", f"Number of frames: {len(frames)}", run_id=run_id)

    # Initialize model
    log_step("Initializing model", run_id=run_id)
    model = init_model()
    log_step("Model initialized", f"Using device: CPU", run_id=run_id)

    # Extract activations
    if not os.path.exists(activations_output_dir) or clean_run:
        activations = perform_activations(model, frames, preprocess_seqn, run_id=run_id)
        save_activations(activations=activations, output_dir=activations_output_dir)
        log_step("Saved activations", activations_output_dir, run_id=run_id)
        print(f"Activations saved in '{activations_output_dir}' directory.")
    else:
        print(f"Activations directory '{activations_output_dir}' already exists. Loading activations...")
        activations = load_activations(activations_output_dir)
        log_step("Loaded existing activations", activations, run_id=run_id)

    # Perform Fourier Transform on activations
    log_step("Performing Fourier Transform", run_id=run_id)
    fourier_transformed_activations = perform_fourier_transform(activations, reduction_method='median')
    log_step("Fourier Transform completed", fourier_transformed_activations, run_id=run_id)

    # Find dominant frequencies
    log_step("Finding dominant frequencies", run_id=run_id)
    dominant_frequencies = find_dominant_frequencies(
        fourier_transformed_activations,
        fps=24,
        threshold_factor=1.5,
        num_peaks=3,
        min_snr=3.0,
        method='two_neighbours'
    )
    log_step("Found dominant frequencies", dominant_frequencies, run_id=run_id)

    # Save dominant frequencies to CSV
    output_csv_path = f'{run_id}_dominant_frequencies.csv'
    log_step("Saving dominant frequencies to CSV", output_csv_path, run_id=run_id)
    save_dominant_frequencies_to_csv(dominant_frequencies, output_csv_path, image_path, gif_frequency1=5, gif_frequency2=6)

    # Skip plotting and saving spectrums to save time
    log_step("Skipping spectrum plots to save time", run_id=run_id)

    return {
        "frames": frames,
        "activations": activations,
        "fourier_transformed_activations": fourier_transformed_activations,
        "dominant_frequencies": dominant_frequencies
    }

def compare_runs(run1_data, run2_data):
    """Compare the results of two runs and identify differences"""
    comparison_file = os.path.join(COMPARISON_DIR, "comparison_results.txt")

    with open(comparison_file, "w") as f:
        f.write("=== COMPARISON RESULTS ===\n\n")

        # Compare frames
        f.write("=== FRAMES COMPARISON ===\n")
        if len(run1_data["frames"]) != len(run2_data["frames"]):
            f.write(f"Different number of frames: Run1 = {len(run1_data['frames'])}, Run2 = {len(run2_data['frames'])}\n")
        else:
            frame_diffs = 0
            for i, (frame1, frame2) in enumerate(zip(run1_data["frames"], run2_data["frames"])):
                if not np.array_equal(frame1, frame2):
                    frame_diffs += 1
                    f.write(f"Frame {i} differs: Sum diff = {np.sum(np.abs(frame1 - frame2))}\n")

            if frame_diffs == 0:
                f.write("All frames are identical\n")
            else:
                f.write(f"{frame_diffs} frames differ out of {len(run1_data['frames'])}\n")

        # Compare activations
        f.write("\n=== ACTIVATIONS COMPARISON ===\n")
        if set(run1_data["activations"].keys()) != set(run2_data["activations"].keys()):
            f.write(f"Different layer keys: Run1 = {sorted(run1_data['activations'].keys())}, Run2 = {sorted(run2_data['activations'].keys())}\n")
        else:
            activation_diffs = 0
            for layer_id in run1_data["activations"]:
                if len(run1_data["activations"][layer_id]) != len(run2_data["activations"][layer_id]):
                    f.write(f"Layer {layer_id}: Different number of frames: Run1 = {len(run1_data['activations'][layer_id])}, Run2 = {len(run2_data['activations'][layer_id])}\n")
                    activation_diffs += 1
                else:
                    layer_diff = False
                    for i, (act1, act2) in enumerate(zip(run1_data["activations"][layer_id], run2_data["activations"][layer_id])):
                        if not np.array_equal(act1, act2):
                            if not layer_diff:
                                f.write(f"Layer {layer_id}: Activations differ\n")
                                layer_diff = True
                            activation_diffs += 1
                            break

            if activation_diffs == 0:
                f.write("All activations are identical\n")
            else:
                f.write(f"Activations differ in {activation_diffs} layers\n")

        # Compare Fourier transformed activations
        f.write("\n=== FOURIER TRANSFORM COMPARISON ===\n")
        if set(run1_data["fourier_transformed_activations"].keys()) != set(run2_data["fourier_transformed_activations"].keys()):
            f.write(f"Different layer keys: Run1 = {sorted(run1_data['fourier_transformed_activations'].keys())}, Run2 = {sorted(run2_data['fourier_transformed_activations'].keys())}\n")
        else:
            fft_diffs = 0
            for layer_id in run1_data["fourier_transformed_activations"]:
                fft1 = run1_data["fourier_transformed_activations"][layer_id]
                fft2 = run2_data["fourier_transformed_activations"][layer_id]

                if fft1.shape != fft2.shape:
                    f.write(f"Layer {layer_id}: Different shapes: Run1 = {fft1.shape}, Run2 = {fft2.shape}\n")
                    fft_diffs += 1
                elif not np.allclose(fft1, fft2, rtol=1e-5, atol=1e-8):
                    max_diff = np.max(np.abs(fft1 - fft2))
                    mean_diff = np.mean(np.abs(fft1 - fft2))
                    f.write(f"Layer {layer_id}: FFT values differ: Max diff = {max_diff}, Mean diff = {mean_diff}\n")
                    fft_diffs += 1

            if fft_diffs == 0:
                f.write("All Fourier transforms are identical\n")
            else:
                f.write(f"Fourier transforms differ in {fft_diffs} layers\n")

        # Compare dominant frequencies
        f.write("\n=== DOMINANT FREQUENCIES COMPARISON ===\n")
        if set(run1_data["dominant_frequencies"].keys()) != set(run2_data["dominant_frequencies"].keys()):
            f.write(f"Different layer keys: Run1 = {sorted(run1_data['dominant_frequencies'].keys())}, Run2 = {sorted(run2_data['dominant_frequencies'].keys())}\n")
        else:
            freq_diffs = 0
            for layer_id in run1_data["dominant_frequencies"]:
                if set(run1_data["dominant_frequencies"][layer_id].keys()) != set(run2_data["dominant_frequencies"][layer_id].keys()):
                    f.write(f"Layer {layer_id}: Different filter keys\n")
                    freq_diffs += 1
                else:
                    for filter_id in run1_data["dominant_frequencies"][layer_id]:
                        freq1 = run1_data["dominant_frequencies"][layer_id][filter_id]
                        freq2 = run2_data["dominant_frequencies"][layer_id][filter_id]

                        if not np.array_equal(freq1, freq2):
                            f.write(f"Layer {layer_id}, Filter {filter_id}: Different frequencies: Run1 = {freq1}, Run2 = {freq2}\n")
                            freq_diffs += 1

            if freq_diffs == 0:
                f.write("All dominant frequencies are identical\n")
            else:
                f.write(f"Dominant frequencies differ in {freq_diffs} filters\n")

    print(f"Comparison results saved to {comparison_file}")
    return comparison_file

def main():
    if len(sys.argv) < 2:
        print("Usage: python run-compare.py <image_path> [clean_run]")
        sys.exit(1)

    image_path = sys.argv[1]
    clean_run = len(sys.argv) > 2 and sys.argv[2].lower() == 'clean'

    print(f"Running comparison on image: {image_path}")
    print(f"Clean run: {clean_run}")

    # Run the pipeline twice
    print("\n=== RUNNING FIRST PASS ===")
    run1_data = run_pipeline(image_path, run_id="run1", clean_run=clean_run)

    print("\n=== RUNNING SECOND PASS ===")
    run2_data = run_pipeline(image_path, run_id="run2", clean_run=clean_run)

    # Compare the results
    print("\n=== COMPARING RESULTS ===")
    comparison_file = compare_runs(run1_data, run2_data)

    print(f"\nComparison complete. Results saved to {comparison_file}")

if __name__ == "__main__":
    main()
