import cv2
import numpy as np
import imageio
import time
import matplotlib.pyplot as plt
import os
from PIL import Image


def flicker_image_and_save_gif(image_path="image.jpg", output_gif="flicker.gif", duration=5, frequency=4, fps=24, color_format="HSV"):
    image = cv2.imread(image_path)

    if image is None:
        print("Error: Could not read the image.")
        return []

    # Initialize frames list
    frames = []

    # Total number of frames
    num_frames = duration * fps
    time_step = np.linspace(0, duration, num_frames)

    # print(f"Generating {num_frames} frames for {duration} seconds at {fps} FPS... in {color_format} format")
    if color_format == "HSV":
        img_color = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        intensity_channel = 2  # V channel (brightness)
        convert_back = cv2.COLOR_HSV2BGR
    elif color_format == "LAB":
        img_color = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        intensity_channel = 0  # L channel (lightness)
        convert_back = cv2.COLOR_LAB2BGR
    elif color_format == "RGB":
        img_color = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        intensity_channel = None  # No specific channel for RGB
        convert_back = None  # No conversion needed
    else:
        print("Error: Unsupported color format.")
        return []

    for t in time_step:
        # Apply sinusoidal flicker effect
        alpha = 0.75 + 0.25 * np.sin(2 * np.pi * frequency * t)  # Sinusoidal flicker range [0.5 - 1.0] so the lowest brightness is 0.5

        # Flickering
        flickered = img_color.copy()
        if intensity_channel is not None:
            flickered[:, :, intensity_channel] = np.clip(img_color[:, :, intensity_channel] * alpha, 0, 255)
        else:
            flickered = np.clip(img_color * alpha, 0, 255).astype(np.uint8)

        if convert_back:
            # Convert back to BGR for visualization (as imageio expects BGR or RGB)
            flickered_image = cv2.cvtColor(flickered, convert_back)
            # Convert to RGB (imageio uses RGB format)
            flickered_image_rgb = cv2.cvtColor(flickered_image, cv2.COLOR_BGR2RGB)
        else:
            flickered_image_rgb = flickered

        # Append frames
        frames.append(flickered_image_rgb)

    # Save the flickered frames as a GIF
    # imageio.mimsave(output_gif, frames, duration=1/fps)

    # print(f"Saved flickering GIF as {output_gif} ({color_format} flickering)")
    return frames

# Example usage:
# flicker_image_and_save_gif(image_path="image.jpg", output_gif="flicker.gif", duration=5, frequency=2, fps=10, color_format="HSV")


def flicker_image_hh_and_save_gif(image_path="image.jpg", output_gif="flicker.gif", duration=5, frequency1=5,frequency2=6, fps=24, color_format="RGB"):
    image = cv2.imread(image_path)#cv2 will read the image as BGR

    if image is None:
        print("Error: Could not read the image.")
        return []

    # Initialize frames list
    frames = []

    # Total number of frames
    num_frames = duration * fps
    time_step = np.linspace(0, duration, num_frames)

    # print(f"Generating {num_frames} frames for {duration} seconds at {fps} FPS... in {color_format} format")
    if color_format == "HSV":
        img_color = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)#conversions are done on BGR
        intensity_channel = 2  # V channel (brightness)
        convert_back = cv2.COLOR_HSV2BGR
    elif color_format == "LAB":
        img_color = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        intensity_channel = 0  # L channel (lightness)
        convert_back = cv2.COLOR_LAB2BGR
    elif color_format == "RGB":
        img_color = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) #conversion is needed from bgr to rgb
        intensity_channel = None  # No specific channel for RGB
        convert_back = None  # No conversion needed
    else:
        print("Error: Unsupported color format.")
        return []
    height, width = img_color.shape[:2]
    mid_col = width // 2

    for t in time_step:
        alpha_left = 0.75 + 0.25 * np.sin(2 * np.pi * frequency1 * t)#range [0.5-1.0] so the lowest brightness is 0.5
        alpha_right = 0.75 + 0.25 * np.sin(2 * np.pi * frequency2 * t)#range [0.5 - 1.0] so the lowest brightness is 0.5

        # Flickering
        flickered = img_color.copy()
        if intensity_channel is not None:
            # Process left half
            flickered[:, :mid_col, intensity_channel] = np.clip(
                img_color[:, :mid_col, intensity_channel] * alpha_left, 0, 255
            )
            # Process right half
            flickered[:, mid_col:, intensity_channel] = np.clip(
                img_color[:, mid_col:, intensity_channel] * alpha_right, 0, 255
            )
        else:
            # For RGB mode
            flickered[:, :mid_col, :] = np.clip(
                img_color[:, :mid_col, :] * alpha_left, 0, 255
            )
            flickered[:, mid_col:, :] = np.clip(
                img_color[:, mid_col:, :] * alpha_right, 0, 255
            )
        if convert_back:
            # Convert back to BGR for visualization (as imageio expects BGR or RGB)
            flickered_image = cv2.cvtColor(flickered, convert_back)
            # Convert to RGB (imageio uses RGB format)
            flickered_image_rgb = cv2.cvtColor(flickered_image, cv2.COLOR_BGR2RGB)
        else:
            flickered_image_rgb = flickered

        # Append frames
        frames.append(flickered_image_rgb)

    # Save the flickered frames as a GIF
    # imageio.mimsave(output_gif, frames, duration=1/fps)

    # print(f"Saved flickering GIF as {output_gif} ({color_format} flickering)")
    return frames

# Example usage:
#flicker_image_hh_and_save_gif(image_path="imgs/Screenshot 2025-03-10 at 15.03.06.png", output_gif="flicker.gif", duration=5, frequency1=5,frequency2=6, fps=10, color_format="RGB")
#flicker_image_and_save_gif(image_path="imgs/Screenshot 2025-03-10 at 15.03.06.png", output_gif="flicker_whole.gif", duration=5, frequency=2, fps=10, color_format="RGB")

def save_frames(frames, frames_dir):
    """
    Save a list of frames as individual image files.

    Args:
        frames (list): List of numpy arrays representing image frames
        frames_dir (str): Directory to save the frames
    """
    os.makedirs(frames_dir, exist_ok=True)
    for i, frame in enumerate(frames):
        frame_image = Image.fromarray(frame.astype(np.uint8))  # Convert numpy array to PIL Image
        frame_path = os.path.join(frames_dir, f"frame_{i}.png")
        frame_image.save(frame_path)
    print(f"Frames saved in '{frames_dir}' directory.")

def load_frames(frames_dir):
    """
    Load frames from a directory of image files.

    Args:
        frames_dir (str): Directory containing frame images

    Returns:
        list: List of numpy arrays representing image frames
    """
    frames = []
    for frame_file in sorted(os.listdir(frames_dir)):
        if frame_file.endswith('.png') or frame_file.endswith('.jpg'):
            frame_path = os.path.join(frames_dir, frame_file)
            frame_image = Image.open(frame_path)
            frames.append(np.array(frame_image))
    return frames