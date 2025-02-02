import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from scipy.io import savemat

import torch
import torch.nn as nn

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

        #u4 = self.up4(d4)
        u3 = self.up3(d4) + d3  # skip
        u2 = self.up2(u3) + d2 # skip
        u1 = self.up1(u2) + d1  # skip

        out = self.final(u1)
        out = self.finalfinal(out)
        return out



class H5Dataset(Dataset):
    def __init__(self, data_path):
        self.data = h5py.File(data_path, 'r')
        self.inputs = self.data['inputs']
        self.outputs = self.data['outputs']
        print(f'H5 Dataset Shape: {self.inputs.shape}')

    def __len__(self):
        return self.outputs.shape[0]  # Number of samples

    def __getitem__(self, idx):
        # Load one sample at a time
        input = self.inputs[idx]
        output = self.outputs[idx]

        # Convert to PyTorch tensors
        input = torch.tensor(input, dtype=torch.float32).unsqueeze(0)  # Add channel dim
        output = torch.tensor(output, dtype=torch.float32).unsqueeze(0)
        return input, output

    def close(self):
        self.data.close()

if __name__ == "__main__":
    # Paths to the .h5 files
    data_path = 'param_data2.hdf5'

    # Load dataset
    dataset = H5Dataset(data_path)

    # Split indices for train and validation
    num_samples = len(dataset)
    train_size = int(0.8 * num_samples)
    val_size = num_samples - train_size

    train_indices, val_indices = train_test_split(range(num_samples), test_size=0.2, random_state=42)

    # Create Subsets for Training and Validation
    train_subset = torch.utils.data.Subset(dataset, train_indices)
    val_subset = torch.utils.data.Subset(dataset, val_indices)

    # Create DataLoaders
    train_loader = DataLoader(train_subset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=32, shuffle=False)


    ### Training
    # Initialize model, loss, optimizer
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = UNet().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.005)

    # Training
    num_epochs = 25
    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        # Training Phase
        model.train()
        train_loss = 0.0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Validation Phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()

        train_losses.append(train_loss / len(train_loader))
        val_losses.append(val_loss / len(val_loader))
        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss / len(train_loader):.4f}, "
              f"Val Loss: {val_loss / len(val_loader):.4f}")

    # Save the trained model
    torch.save(model.state_dict(), 'u_net.pth')

    # Create an index for each epoch
    epochs = list(range(1, len(train_losses) + 1))

    # Plot the training and validation losses
    plt.figure(figsize=(8, 6))
    plt.scatter(epochs, train_losses, label='Training Loss', color='blue', marker='o')
    plt.scatter(epochs, val_losses, label='Validation Loss', color='orange', marker='x')

    # Add labels, title, and legend
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)

    # Show the plot
    plt.show()

    # Close dataset files
    dataset.close()
