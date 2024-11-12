import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
import torchvision.transforms as transforms
from dlsia.core.networks import tunet

output_directory = "../data/output/images_given_with_model"
os.makedirs(output_directory, exist_ok=True)

hyperparams_path = "../dlsia_inpainting_saxs_gisaxs/network_hyperparameters.npy"

# Attempt to load the file with allowing pickle
try:
    hyperparams = np.load(hyperparams_path, allow_pickle=True).item()
    print("Hyperparameters loaded successfully:")
    print(hyperparams)
except Exception as e:
    print(f"Failed to load hyperparameters with np.load: {e}")

    # Attempt to read the file content as text
    try:
        with open(hyperparams_path, 'r') as f:
            content = f.read()
            print("File content:")
            print(content)
    except Exception as e:
        print(f"Failed to read file content: {e}")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# CHANGE: Temporarily set the default tensor type to torch.cuda.FloatTensor
if device.type == 'cuda':
    torch.set_default_tensor_type(torch.cuda.FloatTensor)
    print("Default tensor type set to torch.cuda.FloatTensor")


# Initialize the TUNet model using hyperparameters
tunet_model = tunet.TUNet(
    image_shape=(512, 512),  # Assuming a default image shape, adjust if needed
    in_channels=hyperparams['in_channels'],
    out_channels=hyperparams['out_channels'],
    base_channels=hyperparams.get('base_channels', 4),  # Default to 4 if not specified
    depth=30, # hyperparams.get('num_layers', 3),  # Default to 3 if not specified
    growth_rate=hyperparams.get('growth_rate', 1.5)  # Default to 1.5 if not specified
)

if device.type == 'cuda':
    torch.set_default_tensor_type(torch.FloatTensor)
    print("Default tensor type reset to torch.FloatTensor")

# Load the model weights from the binary file
model_path = "../dlsia_inpainting_saxs_gisaxs/net"
try:
    tunet_model.load_state_dict(torch.load(model_path, map_location=device), strict=False)
    tunet_model.eval()
    print("Model loaded and set to evaluation mode successfully.")
except Exception as e:
    print(f"Failed to load model weights: {e}")
    raise

# Load the input images
input_images_path = "../dlsia_inpainting_saxs_gisaxs/imgs_to_segment.npy"
try:
    input_images = np.load(input_images_path, allow_pickle=True)
    print(f"Input images loaded successfully with shape: {input_images.shape}")
except Exception as e:
    print(f"Failed to load input images: {e}")
    raise

# Convert the numpy array to a PyTorch tensor
preprocessed_images = torch.tensor(input_images, dtype=torch.float32)
print(f"Converted images to tensor with shape: {preprocessed_images.shape}")

# Add batch dimension if necessary
if preprocessed_images.ndim == 3:
    preprocessed_images = preprocessed_images.unsqueeze(0)
    print(f"Added batch dimension, new shape: {preprocessed_images.shape}")

# Process the input images through the model one by one to reduce memory usage
for i in range(preprocessed_images.size(0)):
    single_image = preprocessed_images[i].unsqueeze(0)  # Add batch dimension
    single_image = single_image.to(device)

    try:
        with torch.no_grad():
            output_image = tunet_model(single_image)
        print(f"Processed image {i+1}/{preprocessed_images.size(0)}")
    except Exception as e:
        print(f"Failed to process image {i+1}/{preprocessed_images.size(0)}: {e}")
        continue

    # Post-process and display the first image
    try:
        output_image = output_image.squeeze().cpu().numpy()
        output_image = (output_image - output_image.min()) / (output_image.max() - output_image.min())  # Normalize to [0, 1]
        output_image = (output_image * 255).astype(np.uint8)  # Scale to [0, 255] and convert to uint8
        
        # Convert to PIL Image
        pil_image = Image.fromarray(output_image)
        
        # Display the first image
        if i == 0:
            plt.imshow(pil_image, cmap='gray')
            plt.title('First Output Image')
            plt.axis('off')
            plt.show()
        
        # Save the image
        pil_image.save(os.path.join(output_directory, f"output_{i}.png"))
        print(f"Saved image {i+1}/{preprocessed_images.size(0)} to {output_directory}")

    except Exception as e:
        print(f"Failed to post-process or save image {i+1}/{preprocessed_images.size(0)}: {e}")

print("Processing complete. Output images saved.")

# print(tunet_model)
