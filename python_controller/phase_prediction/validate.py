import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import h5py
import numpy as np
import matplotlib.pyplot as plt
import time
from unet import UNet, HDF5Dataset
from phase_calculator import calculate_phase_mask
from scipy.fft import fft2, fftshift
import jax.numpy as jnp
import jax
from collections import OrderedDict

B = 1
model_path = "best_unet_256_200k.pth"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

dataset = HDF5Dataset("train_data/unet_2000_256.hdf5")
dataloader = data.DataLoader(dataset, batch_size=B, shuffle=True)

# def load_model(model_path):
#     model = UNet(in_channels=1, out_channels=1, init_features=64).to(device)
#     model.load_state_dict(torch.load(model_path))
#     model.eval()
#     return model

def load_model(model_path):
    model = UNet(in_channels=1, out_channels=1, init_features=64).to(device)
    state_dict = torch.load(model_path, map_location=device)

    # Remove "module." prefix if trained with DataParallel
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        new_key = k.replace("module.", "")  # Remove 'module.' prefix
        new_state_dict[new_key] = v

    model.load_state_dict(new_state_dict)
    model.eval()
    return model

def phase_to_intensity(phase_masks_reference):
    def gaussian_beam(mask_size, beam_width):
        x = jnp.linspace(-1, 1, mask_size)
        y = jnp.linspace(-1, 1, mask_size)
        xx, yy = jnp.meshgrid(x, y)
        return jnp.exp(-(xx ** 2 + yy ** 2) / (2 * beam_width ** 2))

    mask_sz = phase_masks_reference.shape[1]

    phase_masks_reference = np.rot90(phase_masks_reference, 1)
    phase_masks_reference = np.flip(phase_masks_reference, 0)
    incident_beam = gaussian_beam(mask_sz, beam_width=.1)

    slm_field = incident_beam * jnp.exp(1j * phase_masks_reference)

    far_field = fftshift(fft2(slm_field)) / ((mask_sz * 2) ** 2)
    intensity = jnp.abs(far_field)

    intensity = np.power(intensity, 1/4)


    return intensity


model = load_model(model_path)
data_iter = iter(dataloader)
inputs, targets = next(data_iter)
inputs, targets = inputs.to(device), targets.to(device)

predictions = model(inputs)

predictions = predictions.detach().cpu().numpy()

criterion = nn.MSELoss()
mse_loss = criterion(torch.tensor(predictions), targets.cpu()).item()
print(f"Mean Squared Error: {mse_loss:.6f}")

i = 0

# Grid size
size = 256
x0, y0 = 160, 132  # Center of Gaussian
sigma = 2  # Spread of Gaussian
A = 1  # Peak amplitude

# Generate 2D coordinate grid
x = np.arange(size)
y = np.arange(size)
X, Y = np.meshgrid(x, y)

# Compute Gaussian function
gaussian = A * np.exp(-((X - x0) ** 2 + (Y - y0) ** 2) / (2 * sigma ** 2))

# Reshape to match input shape (1, 1, 64, 64)
point_intensity_np = gaussian[np.newaxis, np.newaxis, :, :].astype(np.float32)

# Convert to PyTorch tensor
point_intensity = torch.tensor(point_intensity_np, dtype=torch.float32).to(device)
point_prediction = model(point_intensity).detach().cpu().numpy()
point_prediction = np.array(point_prediction[0,0,:,:])

#predicted_far_field = fftshift(fft2(jnp.exp(1j * predictions[i,0,:,:] * np.pi)))
predicted_intensity = phase_to_intensity(predictions[i,0,:,:])

phase_mask = np.array(targets.cpu()[i,0,:,:]) * np.pi
intensity = phase_to_intensity(phase_mask)

intensity_point = phase_to_intensity(point_prediction)

images = [targets.cpu()[i,0,:,:], predictions[i,0,:,:], intensity, predicted_intensity, point_prediction, point_intensity_np[0,0,:,:], intensity_point]
titles = ["Reference Phase", "Predicted Phase", "Reference Intensity", "Predicted Phase Intensity", "Point Prediction", "Point Intensity", "Intensity Predict"]

# Create a row of subplots
fig, axes = plt.subplots(1, len(images), figsize=(15, 5))

for ax, img, title in zip(axes, images, titles):
    im = ax.imshow(img, cmap='gray')
    ax.set_title(title)
    ax.axis('off')  # Hide axes for clarity
    fig.colorbar(im, ax=ax)  # Add individual colorbars

plt.show()
