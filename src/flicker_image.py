import cv2
import numpy as np
import imageio
from PIL import Image

import cv2
import numpy as np
import imageio
import time

def flicker_image_and_save_gif(*, image_path, output_gif="flicker.gif", duration=5, frequency=2, fps=10):
    image = cv2.imread(image_path)
    
    if image is None:
        print("Error: Could not read the image.")
        return
    
    frames = []
    num_frames = duration * fps
    time_step = np.linspace(0, duration, num_frames)

    # HSV flickering
    img_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv_intensity_channel = 2  # V channel (brightness)
    
    # LAB flickering
    img_lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    lab_intensity_channel = 0  # L channel (lightness)

    for t in time_step:
        # Sinusoidal flicker effect for both color spaces
        alpha = 0.5 + 0.5 * np.sin(2 * np.pi * frequency * t)  # Sinusoidal flicker
        
        # Apply the flicker to the intensity channels in both HSV and LAB
        hsv_flickered = img_hsv.copy()
        hsv_flickered[:, :, hsv_intensity_channel] = np.clip(img_hsv[:, :, hsv_intensity_channel] * alpha, 0, 255)

        lab_flickered = img_lab.copy()
        lab_flickered[:, :, lab_intensity_channel] = np.clip(img_lab[:, :, lab_intensity_channel] * alpha, 0, 255)

        # Convert back to BGR for visualization (as imageio expects BGR or RGB)
        flickered_hsv_image = cv2.cvtColor(hsv_flickered, cv2.COLOR_HSV2BGR)
        flickered_lab_image = cv2.cvtColor(lab_flickered, cv2.COLOR_LAB2BGR)

        # Convert to RGB (imageio uses RGB format)
        flickered_hsv_image_rgb = cv2.cvtColor(flickered_hsv_image, cv2.COLOR_BGR2RGB)
        flickered_lab_image_rgb = cv2.cvtColor(flickered_lab_image, cv2.COLOR_BGR2RGB)

        # Append both the HSV and LAB frames to the frames list
        frames.append(flickered_hsv_image_rgb)
        frames.append(flickered_lab_image_rgb)

    # Save the frames as a GIF
    imageio.mimsave(output_gif, frames, duration=1/fps)
    print(f"Saved flickering GIF as {output_gif}")

