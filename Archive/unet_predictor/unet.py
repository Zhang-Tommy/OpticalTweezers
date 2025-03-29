import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import torch.utils.data as data
import h5py
import numpy as np
from collections import OrderedDict
from torch.optim.lr_scheduler import ReduceLROnPlateau


class UNet(nn.Module):

    def __init__(self, in_channels=1, out_channels=1, init_features=32):
        super(UNet, self).__init__()

        features = init_features
        self.encoder1 = UNet._block(in_channels, features, name="enc1")
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder2 = UNet._block(features, features * 2, name="enc2")
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder3 = UNet._block(features * 2, features * 4, name="enc3")
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder4 = UNet._block(features * 4, features * 8, name="enc4")
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bottleneck = UNet._block(features * 8, features * 16, name="bottleneck")

        self.upconv4 = nn.ConvTranspose2d(
            features * 16, features * 8, kernel_size=2, stride=2
        )
        self.decoder4 = UNet._block((features * 8) * 2, features * 8, name="dec4")
        self.upconv3 = nn.ConvTranspose2d(
            features * 8, features * 4, kernel_size=2, stride=2
        )
        self.decoder3 = UNet._block((features * 4) * 2, features * 4, name="dec3")
        self.upconv2 = nn.ConvTranspose2d(
            features * 4, features * 2, kernel_size=2, stride=2
        )
        self.decoder2 = UNet._block((features * 2) * 2, features * 2, name="dec2")
        self.upconv1 = nn.ConvTranspose2d(
            features * 2, features, kernel_size=2, stride=2
        )
        self.decoder1 = UNet._block(features * 2, features, name="dec1")

        self.conv = nn.Conv2d(
            in_channels=features, out_channels=out_channels, kernel_size=1
        )

        self.final_tanh = nn.Tanh()
        self.scale_factor = torch.nn.Parameter(torch.tensor(np.pi), requires_grad=False)

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))

        bottleneck = self.bottleneck(self.pool4(enc4))

        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)
        out = self.conv(dec1)
        out = self.final_tanh(out)  # * self.scale_factor
        return out

    @staticmethod
    def _block(in_channels, features, name):
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "conv1",
                        nn.Conv2d(
                            in_channels=in_channels,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm1", nn.BatchNorm2d(num_features=features)),
                    (name + "relu1", nn.ReLU(inplace=True)),
                    (
                        name + "conv2",
                        nn.Conv2d(
                            in_channels=features,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm2", nn.BatchNorm2d(num_features=features)),
                    (name + "relu2", nn.ReLU(inplace=True)),
                ]
            )
        )

class HDF5Dataset(data.Dataset):
    def __init__(self, file_path):
        self.file_path = file_path
        with h5py.File(self.file_path, 'r') as f:
            self.length = f['inputs'].shape[0]

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        with h5py.File(self.file_path, 'r') as f:
            x = f['inputs'][idx]
            y = f['outputs'][idx]
            y_normalized = y / np.pi
        return torch.tensor(x, dtype=torch.float32).unsqueeze(0), torch.tensor(y_normalized,
                                                                               dtype=torch.float32).unsqueeze(0)

def train_unet(model, train_dataloader, val_dataloader, optimizer, loss_fn, num_epochs=20, patience=10, min_delta=1e-4):
    model.train()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Learning Rate Scheduler (Reduce LR if validation loss plateaus)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

    # Early Stopping Parameters
    best_val_loss = float('inf')
    early_stop_counter = 0

    print(f"Using {torch.cuda.device_count()} GPUs!")

    for epoch in range(num_epochs):
        total_train_loss = 0
        total_val_loss = 0

        model.train()
        for images, masks in train_dataloader:
            images, masks = images.to(device), masks.to(device)

            optimizer.zero_grad()
            outputs = model(images)

            loss = loss_fn(outputs, masks)
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_dataloader)

        model.eval()
        with torch.no_grad():
            for images, masks in val_dataloader:
                images, masks = images.to(device), masks.to(device)

                outputs = model(images)
                loss = loss_fn(outputs, masks)
                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(val_dataloader)
        print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

        # Reduce LR if validation loss plateaus
        scheduler.step(avg_val_loss)

        # Early stopping
        if avg_val_loss + min_delta < best_val_loss:
            best_val_loss = avg_val_loss
            early_stop_counter = 0  # Reset counter if loss improves
            torch.save(model.state_dict(), './best_unet.pth')  # Save best model
        else:
            early_stop_counter += 1
            if early_stop_counter >= patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break

    print("Training finished.")


# Example usage
if __name__ == "__main__":
    batch_sz = 16
    num_epochs = 100
    lr = 0.004
    num_features = 64

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    model = UNet(in_channels=1, out_channels=1, init_features=num_features)
    model = torch.nn.DataParallel(model)
    model = model.to(device)
    dataset = HDF5Dataset("./unet_20000_512.hdf5")
    # dataloader = data.DataLoader(dataset, batch_size=16, shuffle=True)

    dataset_size = len(dataset)
    train_size = int(0.8 * dataset_size)  # 80% training, 20% validation

    val_size = dataset_size - train_size

    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_dataloader = DataLoader(train_dataset, batch_size=batch_sz, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_sz, shuffle=False)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fc = torch.nn.MSELoss()
    # loss_fc = torch.nn.L1Loss()
    train_unet(model, train_dataloader, val_dataloader, optimizer, loss_fc, num_epochs=num_epochs)

    print("Training finished")
