import cv2
import numpy as np
import imageio
from PIL import Image

def flicker_img(img_path, fq1, fq2, output_path, duration=10, fps=30, resize_dim=(640, 480)):
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

        if (i // (fps // fq1)) % 2 == 0: #flicker left half
            frame[:, :width//2] = left_half

        if (i // (fps // fq2)) % 2 == 0: #flicker right half
            frame[:, width//2:] = right_half

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

frames = flicker_img('input_image.jpg', 5, 6, 'output_animation.gif') #example usage
#target_size = (224, 224)  #example target size for CNN input
#preprocessed_frames = [preprocess_frame(frame, target_size) for frame in frames]
