import cv2
import numpy as np
import imageio
import time

def flicker_image_and_save_gif(*, image_path, output_gif_hsv="flicker_hsv.gif", output_gif_lab="flicker_lab.gif", duration=5, frequency=2, fps=10):
    image = cv2.imread(image_path)
    
    if image is None:
        print("Error: Could not read the image.")
        return
    
    # Initialize frames list for both GIFs
    frames_hsv = []
    frames_lab = []
    
    # Total number of frames
    num_frames = duration * fps
    time_step = np.linspace(0, duration, num_frames)

    # Convert image to HSV and LAB color spaces
    img_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv_intensity_channel = 2  # V channel (brightness)
    
    img_lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    lab_intensity_channel = 0  # L channel (lightness)

    for t in time_step:
        # Apply sinusoidal flicker effect to both HSV and LAB color spaces
        alpha = 0.5 + 0.5 * np.sin(2 * np.pi * frequency * t)  # Sinusoidal flicker
        
        # HSV flickering
        hsv_flickered = img_hsv.copy()
        hsv_flickered[:, :, hsv_intensity_channel] = np.clip(img_hsv[:, :, hsv_intensity_channel] * alpha, 0, 255)
        
        # LAB flickering
        lab_flickered = img_lab.copy()
        lab_flickered[:, :, lab_intensity_channel] = np.clip(img_lab[:, :, lab_intensity_channel] * alpha, 0, 255)

        # Convert back to BGR for visualization (as imageio expects BGR or RGB)
        flickered_hsv_image = cv2.cvtColor(hsv_flickered, cv2.COLOR_HSV2BGR)
        flickered_lab_image = cv2.cvtColor(lab_flickered, cv2.COLOR_LAB2BGR)

        # Convert to RGB (imageio uses RGB format)
        flickered_hsv_image_rgb = cv2.cvtColor(flickered_hsv_image, cv2.COLOR_BGR2RGB)
        flickered_lab_image_rgb = cv2.cvtColor(flickered_lab_image, cv2.COLOR_BGR2RGB)

        # Append frames for both GIFs
        frames_hsv.append(flickered_hsv_image_rgb)
        frames_lab.append(flickered_lab_image_rgb)

    # Save the flickered frames as separate GIFs
    imageio.mimsave(output_gif_hsv, frames_hsv, duration=1/fps)
    imageio.mimsave(output_gif_lab, frames_lab, duration=1/fps)

    print(f"Saved flickering GIF as {output_gif_hsv} (HSV flickering)")
    print(f"Saved flickering GIF as {output_gif_lab} (LAB flickering)")

# Example usage:
# flicker_image_and_save_gif(image_path="image.jpg", output_gif_hsv="flicker_hsv.gif", output_gif_lab="flicker_lab.gif", duration=5, frequency=2, fps=10)
