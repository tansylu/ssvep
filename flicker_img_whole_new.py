import cv2
import numpy as np
import imageio
from PIL import Image

def flicker_img(img_path, fq1, output_path, duration=1, fps=30, resize_dim=(640, 480)):
    img = cv2.imread(img_path) #load image
    if img is None:
        raise ValueError("Image not found or unable to load.")
    print(f"Image loaded: {img_path}")

    height, width, _ = img.shape #get image dimensions
    print(f"Image dimensions: {width}x{height}")

    num_frames = int(duration * fps) #calculate number of frames
    frames = []

    for i in range(num_frames):
        frame = np.zeros_like(img)
        #this part is like the previous sabancı research range:0.5-1.0 and sinusoidal function for smoothness.
        # Sinusoidal brightness variation for flickering effect
        # brightness_factor = 0.75 + 0.25 * np.sin(2 * np.pi * fq1 * i / fps)  # Range: 0.5 to 1.0
        brightness_factor = 1
        print(f"Frame {i+1}/{num_frames} - Brightness factor: {brightness_factor}")
        # Apply brightness changes
        frame = np.clip(img * brightness_factor, 0, 255).astype(np.uint8)
        


        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) #convert to RGB
        frame_resized = cv2.resize(frame_rgb, resize_dim) #resize frame
        frames.append(frame_resized) #add to frame list

    print(f"Number of frames generated: {len(frames)}")

    #imageio.mimsave(output_path, frames, fps=fps) #write frames to a gif #this one takes longer
    pil_frames = [Image.fromarray(frame) for frame in frames]
    pil_frames[0].save(output_path, save_all=True, append_images=pil_frames[1:], duration=int(1000/fps), loop=0)
    print(f"GIF saved: {output_path}")

    return frames

def preprocess_frame(frame, target_size):
    frame = cv2.resize(frame, target_size) #resize frame to target size
    frame = frame.astype('float32') / 255.0 #normalize pixel values
    return frame

#frames = flicker_img('/Users/tansylu/Documents/kagglehub/datasets/alxmamaev/flowers-recognition/flowers/dandelion/10477378514_9ffbcec4cf_m.jpg', 5, 6, 'output_animation.gif') #example usage
# target_size = (224, 224)  #example target size for CNN input
# preprocessed_frames = [preprocess_frame(frame, target_size) for frame in frames]
frames = flicker_img('durov.jpg', 5, 'output_animation_whole.gif')