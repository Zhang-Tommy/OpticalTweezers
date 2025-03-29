import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from scipy.io import savemat
from u_net_model import UNet

class H5Dataset(Dataset):
    def __init__(self, far_field_path, phase_mask_path):
        self.far_field_file = h5py.File(far_field_path, 'r')
        self.phase_mask_file = h5py.File(phase_mask_path, 'r')
        self.far_fields = self.far_field_file['/far_fields']
        self.phase_masks = self.phase_mask_file['/phase_masks']

    def __len__(self):
        return self.far_fields.shape[0]  # Number of samples

    def __getitem__(self, idx):
        # Load one sample at a time
        far_field = self.far_fields[idx]
        phase_mask = self.phase_masks[idx]

        # Convert to PyTorch tensors
        far_field = torch.tensor(far_field, dtype=torch.float32).unsqueeze(0)  # Add channel dim
        phase_mask = torch.tensor(phase_mask, dtype=torch.float32).unsqueeze(0)
        return far_field, phase_mask

    def close(self):
        # Close HDF5 files when done
        self.far_field_file.close()
        self.phase_mask_file.close()


if __name__ == "__main__":
    far_field_path = '../../unet/train_data/farfield_full.h5'
    phase_mask_path = '../../unet/train_data/phase_masks_full.h5'

    # Load dataset
    dataset = H5Dataset(far_field_path, phase_mask_path)

    # DataLoader for iterating over samples
    data_loader = DataLoader(dataset, batch_size=3, shuffle=False)

    # Load the trained model
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model = UNet().to(device)
    model.load_state_dict(torch.load('data/phase_predictor_cnn_full.pth'))
    model.eval()

    # Get a few samples for prediction
    inputs, targets = next(iter(data_loader))

    # Move inputs and targets to the same device as the model
    inputs = inputs.to(device)
    targets = targets.to(device)

    # Make predictions
    with torch.no_grad():
        with torch.inference_mode():
            predictions = model(inputs)

    # Move data back to CPU for visualization
    inputs = inputs.cpu().numpy()
    targets = targets.cpu().numpy()
    predictions = predictions.cpu().numpy()

    # Visualize the results
    num_samples = 3  # Number of samples to display
    fig, axes = plt.subplots(num_samples, 3, figsize=(12, 4 * num_samples))

    for i in range(num_samples):
        # Plot Input (Far-Field Intensity)
        axes[i, 0].imshow(inputs[i, 0], cmap='gray')
        axes[i, 0].set_title(f"Input (Far-Field Intensity)")
        axes[i, 0].axis('off')

        # Plot Model Prediction (Phase Mask)
        axes[i, 1].imshow(predictions[i, 0], cmap='gray', aspect='auto', vmin=-0.5, vmax=0.5)
        axes[i, 1].set_title(f"Prediction (Phase Mask)")
        axes[i, 1].axis('off')

        # Plot Ground Truth (Phase Mask)
        axes[i, 2].imshow(targets[i, 0], cmap='gray', aspect='auto', vmin=-0.5, vmax=0.5)
        axes[i, 2].set_title(f"Ground Truth (Phase Mask)")
        axes[i, 2].axis('off')

    plt.tight_layout()
    plt.show()

    # Save predictions, inputs, and targets to .mat files for further analysis
    savemat("data/inputs.mat", {"inputs": inputs[:, 0, :, :]})
    savemat("data/targets.mat", {"targets": targets[:, 0, :, :]})
    savemat("data/predictions.mat", {"predictions": predictions[:, 0, :, :]})

    # Clean up
    dataset.close()
