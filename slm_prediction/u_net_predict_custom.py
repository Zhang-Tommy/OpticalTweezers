import numpy as np
import torch
from scipy.io import loadmat, savemat
from u_net_model import UNet

# Path to the input intensity distribution .mat file and output prediction .mat file
input_mat_path = 'data/custom_gaussian_distribution.mat'  # Replace with your input file name
output_mat_path = 'data/predicted_phase_mask.mat'  # Output file name

# Load the 64x64 intensity distribution from the .mat file
data = loadmat(input_mat_path)
intensity_distribution = data['canvas']  # Replace 'intensity' with the actual key in the .mat file
assert intensity_distribution.shape == (64, 64), "Input array must have shape (64, 64)."

# Convert to PyTorch tensor and add channel and batch dimensions
input_tensor = torch.tensor(intensity_distribution, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

# Load the trained model
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
model = UNet().to(device)
model.load_state_dict(torch.load('data/phase_predictor_cnn_full.pth'))
model.eval()

# Move the input to the same device as the model
input_tensor = input_tensor.to(device)

# Make a prediction
with torch.no_grad():
    prediction = model(input_tensor)

# Move the prediction back to CPU and convert to a NumPy array
prediction = prediction.cpu().numpy().squeeze(0).squeeze(0)  # Remove batch and channel dimensions

# Save the prediction to a .mat file
savemat(output_mat_path, {"predicted_phase_mask": prediction})
print(f"Prediction saved to {output_mat_path}.")
