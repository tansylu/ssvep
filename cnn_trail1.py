## We have not tested this code yet soley for reference.

import keras
from keras.datasets import mnist
import numpy as np
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt

model = ResNet50(weights='imagenet')

layer_names = [layer.name for layer in model.layers if 'conv' in layer.name]

# Create a new model that outputs the selected layers
layer_outputs = [model.get_layer(name).output for name in layer_names]
activation_model = model(inputs=model.input, outputs=layer_outputs)

img_path = 'path/to/your/image.jpg'  # Replace with the path to your image
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

feature_maps = activation_model.predict(x)

for layer_name, feature_map in zip(layer_names, feature_maps):
    num_filters = feature_map.shape[-1]  # Number of filters in this layer
    size = feature_map.shape[1]  # Spatial dimensions of the feature map

    # Plot all filters in this layer
    display_grid = np.zeros((size, size * num_filters))
    
    for i in range(num_filters):
        # Extract ith filter feature map
        x = feature_map[0, :, :, i]
        # Normalize to [0, 1] for visualization
        x -= x.mean()
        x /= (x.std() + 1e-5)
        x *= 64
        x += 128
        x = np.clip(x, 0, 255).astype('uint8')
        display_grid[:, i * size : (i + 1) * size] = x
    
    # Display the feature map grid
    scale = 20. / num_filters
    plt.figure(figsize=(scale * num_filters, scale))
    plt.title(layer_name)
    plt.grid(False)
    plt.imshow(display_grid, aspect='auto', cmap='viridis')
    plt.show()
