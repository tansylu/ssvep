import torch
import torchvision.transforms as transforms
import torchvision.models as models
from flicker_image import flicker_image_and_save_gif

# Load ResNet18 model
print("Loading ResNet18 model...")
resnet18 = models.resnet18()

# Define path to weights file
weights_path = 'resnet18.pth'
print(f"Loading model weights from {weights_path}...")

# Try loading the model weights
try:
    checkpoint = torch.load(weights_path, weights_only=False)  # Allow full loading for legacy formats
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        print("Detected full checkpoint. Extracting model weights...")
        checkpoint = checkpoint['model_state_dict']
    resnet18.load_state_dict(checkpoint)
except Exception as e:
    print(f"Error loading model weights: {e}")
    exit(1)

# Save in a pure weights-only format for future compatibility
torch.save(resnet18.state_dict(), 'resnet18_weights_only.pth')
print("Converted and saved weights-only file: 'resnet18_weights_only.pth'")

# Set model to evaluation mode
print("Setting model to evaluation mode...")
resnet18.eval()
print("Model architecture:")
print(resnet18)

# Define preprocessing transformations
print("Creating preprocessing sequence...")
preprocess_seqn = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Generate flicker image and save as GIF
print("Generating flicker image and saving as GIF...")
frames = flicker_image_and_save_gif(image_path='durov.jpg', frequency=5,  duration=2, fps=30)
print("GIF saved as 'output_animation_whole.gif'.")

# activation_model = ActivationModel(resnet18)
# activations = _get_activations(activation_model, frames)
# save_activations(activations, 'activations_output')
