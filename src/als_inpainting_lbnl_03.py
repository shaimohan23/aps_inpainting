import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from qlty import qlty2D
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


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


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


# Load the model weights from the binary file
model_path = "../dlsia_inpainting_saxs_gisaxs/net"
msdnet_model.load_state_dict(torch.load(model_path, map_location=device), strict=False)
msdnet_model.to(device)
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
    transforms.Resize((512, 512), interpolation=Image.BICUBIC),
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

            input_tensor = input_tensor.to(device)

            # input_image_np = np.array(image)  # Original input image as a NumPy array
            # input_min, input_max = input_image_np.min(), input_image_np.max()  # Get original intensity range
            
            print(f"Loaded and preprocessed image: {filename}")
        except Exception as e:
            print(f"Failed to load or preprocess image {filename}: {e}")
            continue
        
        quilt = qlty2D.NCYXQuilt(X=512,
                                 Y=512,
                                 window=(512, 128),
                                 step=(64, 64),
                                 border=(0, 0),
                                 border_weight=0)

        # Unstitching using qlty
        try:
            input_tensor_cpu = input_tensor.cpu()
            tiles = quilt.unstitch(input_tensor_cpu)
            print(f"Unstitched image into {tiles.shape[0]} tiles")
            print(tiles.shape)
        except Exception as e:
            print(f"Failed during unstitching: {e}")
            continue

        num_passes = 1
        print (f"Number of model iterations per tile in each image: {num_passes}")
        # Process each tile through the model
        outputs = []
        for tile in tiles:
            tile = tile.unsqueeze(0).to(device)  # Add batch dimension
            with torch.no_grad():
                current_input = tile
                for pass_num in range(num_passes):
                    output_tile = msdnet_model(current_input)
                    current_input = output_tile
                outputs.append(output_tile.cpu()) # Move output back to CPU
        
        try:
            # output_tensor = torch.stack(outputs)
            output_tensor = torch.cat(outputs, dim=0)
            print(f"Output tensor shape: {output_tensor.shape}")
            print(f"Output tensor dtype: {output_tensor.dtype}")
            final_output, weights = quilt.stitch(output_tensor)
            print(f"Final output shape: {final_output.shape}")
            print(f"Final output dtype: {final_output.dtype}")
        except Exception as e:
            print(f"Failed during stitching: {e}")
            continue

        try:
            final_output_np = final_output.squeeze().numpy()  # Remove batch dimension and convert to NumPy array
            print(f"Final output shape 2: {final_output.shape}")
            final_output_np = np.clip(final_output_np, 0, None)  # Ensure non-negative

            if final_output_np.ndim > 2:
                final_output_np = final_output_np.squeeze()

            final_output_np = (final_output_np * 255).astype(np.uint8)

            # Create PIL image
            final_image = Image.fromarray(final_output_np, mode='L')  # 'L' mode for grayscale
            
            # Save as PNG instead of JPEG to preserve quality
            # output_path = os.path.splitext(output_path)[0] + '.png'

            # Rescale to original intensity range
            # final_output_np = (final_output_np - final_output_np.min()) / (final_output_np.max() - final_output_np.min())
            # final_output_np = final_output_np * (input_max - input_min) + input_min
            # final_output_np = final_output_np.astype(np.uint8)

            final_image.save(output_path)
        except Exception as e:
            print(f"Failed to post-process or save image {filename}: {e}")
            break
            continue
    break

print("Processing complete. Output images saved.")


