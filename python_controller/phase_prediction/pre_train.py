import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from unet import HDF5Dataset


# Define U-Net with a Pretrained Encoder
class UNetPretrained(nn.Module):
    def __init__(self, encoder_name="resnet34", out_channels=1, init_features=64):
        super(UNetPretrained, self).__init__()

        self.encoder = self.get_pretrained_encoder(encoder_name)

        # Decoder Layers
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, out_channels, kernel_size=1)  # Output layer
        )

    def get_pretrained_encoder(self, name):
        if name == "resnet34":
            model = models.resnet34(pretrained=True)
            layers = list(model.children())[:-2]  # Remove FC layer
        elif name == "resnet50":
            model = models.resnet50(pretrained=True)
            layers = list(model.children())[:-2]
        elif name == "efficientnet_b0":
            model = models.efficientnet_b0(pretrained=True)
            layers = list(model.features.children())  # EfficientNet has feature layers
        elif name == "vgg16":
            model = models.vgg16(pretrained=True)
            layers = list(model.features.children())[:-1]  # Remove classification layers
        else:
            raise ValueError(f"Unsupported encoder: {name}")

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


# Load Data
def get_dataloader(resolution, batch_size):
    file_path = f"./train_data/unet_40000_{resolution}.hdf5"
    dataset = HDF5Dataset(file_path)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    return train_dataloader, val_dataloader


# Training Function
def train_unet(model, train_dataloader, val_dataloader, optimizer, loss_fn, num_epochs=20, patience=10, min_delta=1e-4,
               model_save_path="best_model.pth"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.train()

    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    best_val_loss = float('inf')
    early_stop_counter = 0

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

        scheduler.step(avg_val_loss)

        if avg_val_loss + min_delta < best_val_loss:
            best_val_loss = avg_val_loss
            early_stop_counter = 0
            torch.save(model.state_dict(), model_save_path)
        else:
            early_stop_counter += 1
            if early_stop_counter >= patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break

    print("Training finished.")


# Main Script
if __name__ == "__main__":
    resolutions = [64]
    batch_sz = 16
    num_epochs = 100
    lr = 0.0001
    num_features = 64
    model_save_path = "best_unet_64.pth"
    encoder_choice = "resnet34"  # Change this to "resnet50", "efficientnet_b0", or "vgg16"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNetPretrained(encoder_name=encoder_choice, out_channels=1, init_features=num_features).to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fc = torch.nn.MSELoss()

    for res in resolutions:
        print(f"\n===== Training at {res}x{res} Resolution with {encoder_choice} Encoder =====")
        train_dataloader, val_dataloader = get_dataloader(res, batch_sz)

        try:
            model.load_state_dict(torch.load(model_save_path))
            print(f"Loaded pretrained model for {res}x{res}")
        except FileNotFoundError:
            print(f"No previous model found, starting fresh for {res}x{res}")

        train_unet(model, train_dataloader, val_dataloader, optimizer, loss_fc, num_epochs=num_epochs,
                   model_save_path=model_save_path)

    print("Training finished.")
