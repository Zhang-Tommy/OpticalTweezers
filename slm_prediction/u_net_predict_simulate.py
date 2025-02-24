import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from scipy.io import savemat
#from u_net_model import UNet
from u_net_predict import H5Dataset
import time

class UNet(torch.nn.Module):
    def __init__(self, input_channels=1, output_channels=1, base_channels=64): # 1->64 2->128
        super(UNet, self).__init__()

        # encoding layers
        self.down1 = self.enc(input_channels, base_channels)
        self.down2 = self.enc(base_channels, base_channels * 2)
        self.down3 = self.enc(base_channels * 2, base_channels * 4)
        self.down4 = self.enc(base_channels * 4, base_channels * 8)
        self.down5 = self.enc(base_channels * 8, base_channels * 16)

        # decoding layers
        self.up4 = self.dec(base_channels * 16, base_channels * 8)
        self.up3 = self.dec(base_channels * 8, base_channels * 4)
        self.up2 = self.dec(base_channels * 4, base_channels * 2)
        self.up1 = self.dec(base_channels * 2, base_channels)

        self.final = nn.Conv2d(base_channels, output_channels, kernel_size=1)

        self.finalfinal = nn.Tanh()

        # MaxPool for downsampling
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def enc(self, in_channels, out_channels):
        """Convolutional block: Conv -> BatchNorm -> ReLU."""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def dec(self, in_channels, out_channels):
        """Upsampling block: Transposed Conv -> Conv Block."""
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(self.pool(d1))
        d3 = self.down3(self.pool(d2))
        d4 = self.down4(self.pool(d3))

        u3 = self.up3(d4) + d3  # skip
        u2 = self.up2(u3) + d2 # skip
        u1 = self.up1(u2) + d1  # skip

        out = self.final(u1)
        out = self.finalfinal(out)
        return out

if __name__ == "__main__":
    far_field_path = 'data/farfield_full.h5'
    phase_mask_path = 'data/phase_masks_full.h5'

    # Number of samples to predict on
    num_samples = 1

    # Load dataset
    dataset = H5Dataset(far_field_path, phase_mask_path)

    # Ensure we only take `num_samples` from the dataset
    selected_dataset = torch.utils.data.Subset(dataset, range(num_samples))

    # DataLoader for iterating over samples
    data_loader = DataLoader(selected_dataset, batch_size=num_samples, shuffle=False)

    # Load the trained model
    #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    print(f"Using device: {device}")
    model = UNet().to(device)
    model.load_state_dict(torch.load('data/phase_predictor_cnn_full.pth'))
    model.eval()

    # Get the selected samples for prediction
    inputs, targets = next(iter(data_loader))

    # Move inputs and targets to the same device as the model
    inputs = inputs.to(device)
    targets = targets.to(device)

    start_time = time.time()
    # Make predictions
    with torch.no_grad():
        predictions = model(inputs)
    end_time = time.time()

    inference_time = end_time - start_time
    print(f"Inference time for {num_samples} samples: {inference_time:.4f} seconds")
    # Move data back to CPU for visualization and saving
    inputs = inputs.cpu().numpy()
    targets = targets.cpu().numpy()
    predictions = predictions.cpu().numpy()

    # Save predictions to .mat files
    savemat("data/inputs_simulate.mat", {"inputs": inputs})
    savemat("data/targets_simulate.mat", {"targets": targets})
    savemat("data/predictions_simulate.mat", {"predictions": predictions})

    # Clean up
    dataset.close()
