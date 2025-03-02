import cv2
import numpy as np
import imageio
import time

def flicker_image_and_save_gif(*, image_path, output_gif="flicker.gif", duration=5, frequency=2, fps=10, color_format="HSV") -> list:
    image = cv2.imread(image_path)
    
    if image is None:
        print("Error: Could not read the image.")
        return []
    
    # Initialize frames list
    frames = []
    
    # Total number of frames
    num_frames = duration * fps
    time_step = np.linspace(0, duration, num_frames)

    print(f"Generating {num_frames} frames for {duration} seconds at {fps} FPS... in {color_format} format")
    if color_format == "HSV":
        img_color = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        intensity_channel = 2  # V channel (brightness)
        convert_back = cv2.COLOR_HSV2BGR
    elif color_format == "LAB":
        img_color = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        intensity_channel = 0  # L channel (lightness)
        convert_back = cv2.COLOR_LAB2BGR
    elif color_format == "RGB":
        img_color = image
        intensity_channel = None  # No specific channel for RGB
        convert_back = None  # No conversion needed
    else:
        print("Error: Unsupported color format.")
        return []

    for t in time_step:
        # Apply sinusoidal flicker effect
        alpha = 0.5 + 0.5 * np.sin(2 * np.pi * frequency * t)  # Sinusoidal flicker
        
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
    imageio.mimsave(output_gif, frames, duration=1/fps)

    print(f"Saved flickering GIF as {output_gif} ({color_format} flickering)")
    return frames

# Example usage:
# flicker_image_and_save_gif(image_path="image.jpg", output_gif="flicker.gif", duration=5, frequency=2, fps=10, color_format="HSV")