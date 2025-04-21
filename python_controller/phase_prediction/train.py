import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, random_split
from unet import UNet, HDF5Dataset
import torch.nn.functional as F
import numpy as np
from scipy.fft import fft2, fftshift

def get_dataloader(resolution, batch_size):
    file_path = f"./train_data/unet_200000_{resolution}.hdf5"
    dataset = HDF5Dataset(file_path)
    train_size = int(0.8 * len(dataset))  # 80% training, 20% validation
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    return train_dataloader, val_dataloader

def train_unet(model, train_dataloader, val_dataloader, optimizer, loss_fn, num_epochs=20, patience=10, min_delta=1e-4, model_save_path="best_model.pth"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.train()

    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

    best_val_loss = float('inf')
    early_stop_counter = 0

    for epoch in range(num_epochs):
        total_train_loss = 0
        total_val_loss = 0

        # Training
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

        # Validation
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

        # Early Stopping
        if avg_val_loss + min_delta < best_val_loss:
            best_val_loss = avg_val_loss
            early_stop_counter = 0
            torch.save(model.state_dict(), model_save_path)  # Save best model
        else:
            early_stop_counter += 1
            if early_stop_counter >= patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break

    print("Training finished.")

if __name__ == "__main__":
    resolutions = [128]
    batch_sz = 16
    num_epochs = 100
    lr = 0.0001
    num_features = 64
    model_save_path = "best_unet_128_200k_corrected_10200.pth"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet(in_channels=1, out_channels=1, init_features=num_features).to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fc = torch.nn.MSELoss()
    #loss_fc = IntensityLoss(mask_size=64).to(device)
    # Progressive Training Loop
    for res in resolutions:
        print(f"\n===== Training at {res}x{res} Resolution =====")
        train_dataloader, val_dataloader = get_dataloader(res, batch_sz)

        # Load previous model if available (helps with progressive learning)
        if res > 64:
            try:
                model.load_state_dict(torch.load(model_save_path))
                print(f"Loaded pretrained model for {res}x{res}")
            except FileNotFoundError:
                print(f"No previous model found, starting fresh for {res}x{res}")

        train_unet(model, train_dataloader, val_dataloader, optimizer, loss_fc, num_epochs=num_epochs, model_save_path=model_save_path)

    print("Progressive training finished.")
