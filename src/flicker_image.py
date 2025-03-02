import cv2
import numpy as np
import imageio
from PIL import Image

def flicker_img_from_path(*, img_path, frequency, output_path, duration=1, fps=30, ):
    img = cv2.imread(img_path) #load image
    if img is None:
        raise ValueError("Image not found or unable to load.")
    print(f"Image loaded: {img_path}")

    height, width, _ = img.shape #get image dimensions
    print(f"Image dimensions: {width}x{height}")

    left_half = img[:, :width//2] #split image in vertically half
    right_half = img[:, width//2:]

    num_frames = int(duration * fps) #calculate number of frames
    frames = []

    for i in range(num_frames):
        frame = np.zeros_like(img)

        #this flickering process is the same as previous sabancı research.
        #sinusoidal flickering of brightness from 0.5 to 1 
        brightness_factor_left= 0.75 + 0.25 * np.sin(2*np.pi*frequency*i/fps) 
        brightness_factor_right= 0.75 + 0.25 * np.sin(2*np.pi*frequency*i/fps)

        #apply changes
        frame[:, :width // 2] = np.clip(left_half * brightness_factor_left, 0, 255).astype(np.uint8)
        frame[:, width // 2:] = np.clip(right_half * brightness_factor_right, 0, 255).astype(np.uint8)

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) #convert to RGB
        frames.append(frame_resized) #add to frame list

    print(f"Number of frames generated: {len(frames)}")

    #imageio.mimsave(output_path, frames, fps=fps) #write frames to a gif #this one takes longer
    pil_frames = [Image.fromarray(frame) for frame in frames]
    pil_frames[0].save(output_path, save_all=True, append_images=pil_frames[1:], duration=int(1000/fps), loop=0)
    print(f"GIF saved: {output_path}")

    return frames
