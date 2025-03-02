import torch
import torchvision.transforms as transforms
import torchvision.models as models
from flicker_image import flicker_img_from_path

resnet18 = models.resnet18() 

weights_path = 'resnet18.pth'
resnet18.load_state_dict(torch.load(weights_path)) #load pre-trained model weights

resnet18.eval() #set model to evaluation mode
print(resnet18) #verify model architecture

# Sequence of preprocessing transformations for an image:
# Resize the image to 224x224 pixels.
# Convert the image into a tensor.
# Normalize the tensor by subtracting the mean and dividing by the standard deviation for each channel, using values from the ImageNet dataset.

preprocess_seqn = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(), #for converting the PIL Image to a tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

frames = flicker_img_from_path(img_path='durov.jpg', frequency=5,  output_path='output_animation_whole.gif', duration=2, fps=30)
# activation_model = ActivationModel(resnet18)
# activations = _get_activations(activation_model, frames)
# save_activations(activations, 'activations_output')