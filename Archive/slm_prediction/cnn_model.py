import scipy
import numpy as np
from scipy import io
from scipy.io import loadmat
import mat73
intensity = mat73.loadmat(r'C:\Users\tommyz\Downloads\otslm-1.0.1\otslm-1.0.1\farfield.mat')
phase = mat73.loadmat(r'C:\Users\tommyz\Downloads\otslm-1.0.1\otslm-1.0.1\phase_masks.mat')

far_fields = np.asarray(intensity.get('far_fields'))
phase_patterns = np.asarray(phase.get('phase_masks'))
from scipy.io import savemat
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
torch.set_printoptions(precision=10)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

# Assuming far_fields and phase_patterns are already loaded as 512x512xN arrays
far_fields = torch.tensor(far_fields, dtype=torch.float32, device=device)
phase_patterns = torch.tensor(phase_patterns, dtype=torch.float32, device=device)

far_fields = far_fields.permute(2, 0, 1).unsqueeze(1)  # Shape: [N, 1, 512, 512]
phase_patterns = phase_patterns.permute(2, 0, 1).unsqueeze(1)  # Shape: [N, 1, 512, 512]

train_far_fields, val_far_fields, train_phase_patterns, val_phase_patterns = train_test_split(
    far_fields, phase_patterns, test_size=0.2, random_state=42
)

# Create DataLoader for batching
train_dataset = TensorDataset(train_far_fields, train_phase_patterns)
val_dataset = TensorDataset(val_far_fields, val_phase_patterns)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        # Encoder: progressively reduce dimensions with stride=2
        kern_sz = 3
        strd = 1
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=kern_sz, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Decoder: upsample back to the original size
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(16, 1,
                               kernel_size=kern_sz,
                               stride=2,
                               padding=1,
                               output_padding=1),
            nn.ReLU()
        )

    def forward(self, x):
        #print("Before encoder, shape:", x.shape)
        x = self.encoder(x)
        #print("After encoder, shape:", x.shape)  # Check encoder output
        x = self.decoder(x)
        #print("After decoder, shape:", x.shape)  # Check final output
        return x


class PhasePredictorCNN(nn.Module):
    def __init__(self):
        super(PhasePredictorCNN, self).__init__()

        # Encoder: progressively reduce dimensions with stride=2
        kern_sz = 3
        strd = 1
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=kern_sz, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 8, kernel_size=kern_sz, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Decoder: upsample back to the original size
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(8, 16,
                               kernel_size=kern_sz,
                               stride=2,
                               padding=1,
                               output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1,
                               kernel_size=kern_sz,
                               stride=2,
                               padding=1,
                               output_padding=1),
            nn.ReLU(),
            nn.Tanh()
        )

    def forward(self, x):
        #print("Before encoder, shape:", x.shape)
        x = self.encoder(x)
        #print("After encoder, shape:", x.shape)  # Check encoder output
        x = self.decoder(x)
        #print("After decoder, shape:", x.shape)  # Check final output
        return x

class PhaseCNN(nn.Module):
    def __init__(self):
        super(PhaseCNN, self).__init__()

        # Encoder
        self.enc1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        )
        self.enc3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        )

        # Decoder
        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU()
        )
        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU()
        )
        self.out = nn.Sequential(
            nn.Conv2d(16, 1, kernel_size=3, padding=1),
            nn.Tanh()  # Output range [-1, 1]
        )

    def forward(self, x):
        # Encoder
        enc1_out = self.enc1(x)  # [B, 16, 32, 32]
        enc2_out = self.enc2(enc1_out)  # [B, 32, 16, 16]
        enc3_out = self.enc3(enc2_out)  # [B, 64, 8, 8]

        # Decoder
        dec1_out = self.dec1(enc3_out)  # [B, 32, 16, 16]
        dec2_out = self.dec2(dec1_out + enc2_out)  # Add skip connection, [B, 16, 32, 32]
        out = self.out(dec2_out + enc1_out)  # Add skip connection

        return out


class UNetWithSkipConnections(nn.Module):
    def __init__(self):
        super(UNetWithSkipConnections, self).__init__()

        # Encoder
        self.enc1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),  # Downsampling
            nn.ReLU()
        )
        self.enc3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # Further Downsampling
            nn.ReLU()
        )

        # Decoder
        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU()
        )
        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(64, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            # Skip connection doubles channels
            nn.ReLU()
        )
        self.out = nn.Sequential(
            nn.Conv2d(32, 1, kernel_size=3, padding=1),  # Skip connection doubles channels
            nn.Tanh()  # Output range [-1, 1]
        )

    def forward(self, x):
        # Encoder
        enc1_out = self.enc1(x)  # [B, 16, 32, 32]
        enc2_out = self.enc2(enc1_out)  # [B, 32, 16, 16]
        enc3_out = self.enc3(enc2_out)  # [B, 64, 8, 8]

        # Decoder
        dec1_out = self.dec1(enc3_out)  # [B, 32, 16, 16]
        dec1_out = torch.cat((dec1_out, enc2_out),
                             dim=1)  # Skip connection: Concatenate along channel axis [B, 64, 16, 16]

        dec2_out = self.dec2(dec1_out)  # [B, 16, 32, 32]
        dec2_out = torch.cat((dec2_out, enc1_out),
                             dim=1)  # Skip connection: Concatenate along channel axis [B, 32, 32, 32]

        out = self.out(dec2_out)  # [B, 1, 32, 32]

        return out


class UNet(torch.nn.Module):
    def __init__(self, input_channels=1, output_channels=1, base_channels=16):
        super(UNet, self).__init__()

        # Downsampling layers
        self.down1 = self._conv_block(input_channels, base_channels)
        self.down2 = self._conv_block(base_channels, base_channels * 2)
        self.down3 = self._conv_block(base_channels * 2, base_channels * 4)

        # Upsampling layers
        self.up2 = self._up_block(base_channels * 4, base_channels * 2)
        self.up1 = self._up_block(base_channels * 2, base_channels)

        # Final convolution to match output size
        self.final = nn.Conv2d(base_channels, output_channels, kernel_size=1)
        self.finalfinal = nn.Tanh()
        # MaxPool for downsampling
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def _conv_block(self, in_channels, out_channels):
        """Convolutional block: Conv -> BatchNorm -> ReLU."""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def _up_block(self, in_channels, out_channels):
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
        # Downsample
        d1 = self.down1(x)  # First downsampling layer
        d2 = self.down2(self.pool(d1))  # Second downsampling layer
        d3 = self.down3(self.pool(d2))  # Third downsampling layer

        # Upsample
        u2 = self.up2(d3) + d2  # Skip connection
        u1 = self.up1(u2) + d1  # Skip connection

        # Final output
        f_out = self.final(u1)
        out = self.finalfinal(f_out)
        return out



# Initialize the model, loss function, and optimizer
model = UNet().to(device)
criterion = nn.MSELoss()  # Mean Squared Error for regression
optimizer = optim.Adam(model.parameters(), lr=0.005)

# Training loop
num_epochs = 250
for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    for inputs, targets in train_loader:
        #print(inputs)
        #print(f"Target: {targets}")
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    # Validation loop
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            #print(inputs)
            val_loss += loss.item()

    # Print epoch statistics
    print(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss / len(train_loader):.4f}, "
          f"Validation Loss: {val_loss / len(val_loader):.4f}")

# Save the trained model
torch.save(model.state_dict(), 'data/phase_predictor_cnn.pth')

import matplotlib.pyplot as plt

# Get a batch of validation data
inputs, targets = next(iter(val_loader))
inputs, targets = inputs.to(device), targets.to(device)

# Set the model to evaluation mode and get predictions
model.eval()
with torch.no_grad():
    predictions = model(inputs)

# Move data back to CPU for plotting
inputs = inputs.cpu().numpy()
targets = targets.cpu().numpy()
predictions = predictions.cpu().numpy()

# Plotting: show the input, prediction, and target
num_samples = 3  # Number of examples to display
fig, axes = plt.subplots(num_samples, 3, figsize=(12, 4 * num_samples))

print(inputs.shape)


savemat("C:/Users/tommyz/Downloads/otslm-1.0.1/otslm-1.0.1/inputs.mat", {"inputs": inputs[:,0,:,:]})
savemat("C:/Users/tommyz/Downloads/otslm-1.0.1/otslm-1.0.1/targets.mat", {"targets": targets[:,0,:,:]})
savemat("C:/Users/tommyz/Downloads/otslm-1.0.1/otslm-1.0.1/predictions.mat", {"predictions": predictions[:,0,:,:]})
for i in range(num_samples):
    # Plot Input
    axes[i, 0].imshow(inputs[i, 0], cmap='gray')
    axes[i, 0].set_title(f"Input (Far-Field Intensity)")
    axes[i, 0].axis('off')
    #print(f"Input = {inputs[i, 0]}")

    # Plot Model Prediction
    axes[i, 1].imshow(predictions[i, 0], cmap='gray', aspect='auto', vmin=-0.5, vmax=0.5)
    axes[i, 1].set_title(f"Prediction (Phase Mask)")
    axes[i, 1].axis('off')
    pred = predictions[i, 0]
    #print(f"Model Prediction = {predictions[i, 0]}")

    # Plot Ground Truth
    axes[i, 2].imshow(targets[i, 0], cmap='gray', aspect='auto', vmin=-0.5, vmax=0.5)
    axes[i, 2].set_title(f"Ground Truth (Phase Mask)")
    axes[i, 2].axis('off')
    targ = targets[i, 0]
    #print(f"Ground Truth = {targets[i, 0]}")

plt.tight_layout()
plt.show()