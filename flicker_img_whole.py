import cv2
import numpy as np
import imageio
from PIL import Image
import os
def create_folder_gif(folder_path, fq1,fq2, output_path, image_duration=30, fps=30, resize_dim=(640,480)):
    """
    Processes all images in the specified folder, applies the flicker effect to each for a
    given duration (default 30 seconds), and compiles them into a single GIF.
    """
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp','.JPG')
    # Get a sorted list of image file paths from the folder.
    image_files = [os.path.join(folder_path, f)
                   for f in os.listdir(folder_path)
                   if f.lower().endswith(valid_extensions)]
    image_files.sort()
    print(f"Found {len(image_files)} image files.")

    all_frames = []
    for img_path in image_files:
        print(f"Processing image: {img_path}")
        # Generate flicker frames for each image; don't save individually (output_path=None)
        frames = flicker_img(img_path, fq1,fq2, output_path=None,
                             duration=image_duration, fps=fps, resize_dim=resize_dim)
        
        all_frames.extend(frames)

    if not all_frames:
        raise ValueError("No frames generated. Check if folder contains valid image files.")

    # Save the combined frames as one final GIF
    pil_frames = [Image.fromarray(frame) for frame in all_frames]
    pil_frames[0].save(output_path, format='GIF', save_all=True, 
                     append_images=pil_frames[1:], duration=int(1000/fps), loop=0)
    print(f"Final GIF saved: {output_path}")
    return all_frames

def flicker_img(img_path, fq1,fq2, output_path=None, duration=1, fps=30, resize_dim=(640, 480)):
    """
    Applies a sinusoidal brightness flicker to the image at img_path.
    If output_path is provided and valid, the flickered animation is saved as a GIF.
    Returns the list of generated frames.
    """
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError("Image not found or unable to load.")
    print(f"Image loaded: {img_path}")

    height, width, _ = img.shape
    print(f"Image dimensions: {width}x{height}")

    left_half = img[:, :width//2] #split image in vertically half
    right_half = img[:, width//2:]

    num_frames = int(duration * fps) #calculate number of frames
    frames = []

    for i in range(num_frames):
        frame = np.zeros_like(img)

        brightness_factor_left= 0.75 + 0.25 * np.sin(2*np.pi*fq1*i/fps) 
        brightness_factor_right= 0.75 + 0.25 * np.sin(2*np.pi*fq2*i/fps)

        #apply changes
        frame[:, :width // 2] = np.clip(left_half * brightness_factor_left, 0, 255).astype(np.uint8)
        frame[:, width // 2:] = np.clip(right_half * brightness_factor_right, 0, 255).astype(np.uint8)

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) #convert to RGB
        frame_resized = cv2.resize(frame_rgb, resize_dim) #resize frame

        frames.append(frame_resized)

    print(f"Number of frames generated: {len(frames)}")

    # Save the GIF only if a valid output_path is provided
    if output_path is not None and os.path.splitext(output_path)[1].lower() == ".gif":
        pil_frames = [Image.fromarray(frame) for frame in frames]
        pil_frames[0].save(output_path, save_all=True, append_images=pil_frames[1:],
                           duration=int(1000/fps), loop=0)
        print(f"GIF saved: {output_path}")
    elif output_path is not None:
        raise ValueError(f"Output path must have a .gif extension. Provided: {output_path}")

    return frames


def preprocess_frame(frame, target_size):
    frame = cv2.resize(frame, target_size) #resize frame to target size
    frame = frame.astype('float32') / 255.0 #normalize pixel values
    return frame

#frames = flicker_img('/Users/tansylu/Documents/kagglehub/datasets/alxmamaev/flowers-recognition/flowers/dandelion/10477378514_9ffbcec4cf_m.jpg', 5, 6, 'output_animation.gif') #example usage
# target_size = (224, 224)  #example target size for CNN input
# preprocessed_frames = [preprocess_frame(frame, target_size) for frame in frames]
#frames = flicker_img('durov.jpg', 5, 'output_animation_whole.gif')

#folder_path = "imgs"
#output_gif = "output_animation2.gif"
#create_folder_gif(folder_path, fq1=5, output_path=output_gif, image_duration=1, fps=30, resize_dim=(640,480))