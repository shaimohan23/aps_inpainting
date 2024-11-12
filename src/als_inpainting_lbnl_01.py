import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
import torchvision.transforms as transforms
from dlsia.core.networks import tunet, msdnet

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


# CHANGE: Check if CUDA is available and set the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# CHANGE: Temporarily set the default tensor type to torch.cuda.FloatTensor
if device.type == 'cuda':
    torch.set_default_tensor_type(torch.cuda.FloatTensor)
    print("Default tensor type set to torch.cuda.FloatTensor")


# Ensure you have the tunet module imported
# import tunet  # Uncomment if tunet is a module you have

torch.cuda.empty_cache()
# Initialize the TUNet model using hyperparameters
# tunet_model = tunet.TUNet(
#     image_shape=(512, 512),
#     in_channels=hyperparams['in_channels'],
#     out_channels=hyperparams['out_channels'],
#     base_channels=hyperparams.get('base_channels', 4),
#     depth= 30, # hyperparams['depth'],
#     growth_rate=hyperparams.get('growth_rate', 1.5)
# )

msdnet_model = msdnet.MixedScaleDenseNetwork(
    in_channels=hyperparams['in_channels'],
    out_channels=hyperparams['out_channels'],
    num_layers= hyperparams['num_layers'],
)

# CHANGE: Reset default tensor type back to torch.FloatTensor
if device.type == 'cuda':
    torch.set_default_tensor_type(torch.FloatTensor)
    print("Default tensor type reset to torch.FloatTensor")


# Load the model weights from the binary file
model_path = "../dlsia_inpainting_saxs_gisaxs/net"
# state_dict = torch.load(model_path, map_location=device)
# print (state_dict)
msdnet_model.load_state_dict(torch.load(model_path, map_location=device), strict=False)
msdnet_model.eval()
print("Model loaded and set to evaluation mode successfully.")

# Directory of input images
input_directory = "../data/input/LBNL_ALS_Scattering_data_jpg"

# Directory to save the output images
output_directory = "../data/output/LBNL_ALS_Scattering_data_jpg_inpainted"
os.makedirs(output_directory, exist_ok=True)

# Define image transformations
def safe_divide(x):
    max_val = x.max()
    if max_val > 0:
        return x / max_val
    else:
        return x

def safe_log1p(x):
    return torch.log1p(torch.clamp(x, min=0))

transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Lambda(safe_divide),
    transforms.Lambda(safe_log1p)
])

# Process each image in the directory
for filename in os.listdir(input_directory):
    if filename.lower().endswith(('.jpg', '.jpeg', '.tiff', '.png')):
        image_path = os.path.join(input_directory, filename)
        output_path = os.path.join(output_directory, filename)

        # Load and preprocess the image
        try:
            image = Image.open(image_path).convert("L")  # Convert to grayscale
            input_tensor = transform(image).unsqueeze(0)  # Add batch dimension

            # CHANGE: Move the input tensor to the device
            input_tensor = input_tensor.to(device)

            input_image_np = np.array(image)  # Original input image as a NumPy array
            input_min, input_max = input_image_np.min(), input_image_np.max()  # Get original intensity range
            
            print(f"Loaded and preprocessed image: {filename}")
        except Exception as e:
            print(f"Failed to load or preprocess image {filename}: {e}")
            continue
        

        # Run the model
        try:
            with torch.no_grad():
                for _ in range(3):
                    output_tensor = msdnet_model(input_tensor)
                    input_tensor = output_tensor
            print(f"Processed image: {filename}")
        except Exception as e:
            print(f"Failed to process image {filename}: {e}")
            continue
        
        # Post-process and save the output
        try:
            # output_image = output_tensor.squeeze(0).squeeze(0).cpu().numpy()
            # # Normalize output_image to [0, 255]
            # output_image = (output_image - output_image.min()) / (output_image.max() - output_image.min())
            
            # contrast_factor = 1.5  # Adjust this value as needed
            # output_image = np.clip((output_image - 0.5) * contrast_factor + 0.5, 0, 1)
            
            # output_image = (output_image * 255).astype(np.uint8)

            output_image = output_tensor.squeeze(0).squeeze(0).cpu().numpy()
            # Normalize the output image to [0, 1]
            output_image = (output_image - output_image.min()) / (output_image.max() - output_image.min())
            # Scale the output image back to the original intensity range
            output_image = output_image * (input_max - input_min) + input_min
            output_image = np.clip(output_image, 0, 255).astype(np.uint8)

            # Convert to PIL Image and save
            output_pil_image = Image.fromarray(output_image)
            output_pil_image.save(output_path)
            print(f"Saved output image: {output_path}")
        except Exception as e:
            print(f"Failed to post-process or save image {filename}: {e}")
            continue

print("Processing complete. Output images saved.")


