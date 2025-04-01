import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, random_split
from unet import UNet, HDF5Dataset, FH5Dataset
import torch.nn.functional as F

def train_unet(model, train_dataloader, val_dataloader, optimizer, loss_fn, num_epochs=20, patience=10, min_delta=1e-4):
    model.train()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Learning Rate Scheduler (Reduce LR if validation loss plateaus)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

    # Early Stopping Parameters
    best_val_loss = float('inf')
    early_stop_counter = 0

    #print(f"Using {torch.cuda.device_count()} GPUs!")

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
            torch.save(model.state_dict(), 'old_files/best_unet_256.pth')  # Save best model
        else:
            early_stop_counter += 1
            if early_stop_counter >= patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break

    print("Training finished.")

# def train_unet(model, train_dataloader, val_dataloader, optimizer, loss_fn, num_epochs=20):
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#
#     for epoch in range(num_epochs):
#         total_train_loss = 0
#         total_val_loss = 0
#
#         model.train()
#         for images, masks in train_dataloader:
#             images, masks = images.to(device), masks.to(device)
#
#             optimizer.zero_grad()
#             outputs = model(images)
#
#             loss = loss_fn(outputs, masks)
#             loss.backward()
#             optimizer.step()
#
#             total_train_loss += loss.item()
#
#         avg_train_loss = total_train_loss / len(train_dataloader)
#
#         model.eval()
#         with torch.no_grad():
#             for images, masks in val_dataloader:
#                 images, masks = images.to(device), masks.to(device)
#
#                 outputs = model(images)
#                 loss = loss_fn(outputs, masks)
#                 total_val_loss += loss.item()
#
#         avg_val_loss = total_val_loss / len(val_dataloader)
#         print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
#         torch.save(model.state_dict(), './best_unet_64.pth')  # Save best model
#     print("Training finished.")

def gradient_loss(pred, target):
    """Computes the gradient loss to preserve phase details."""
    grad_pred_x = pred[:, :, :, :-1] - pred[:, :, :, 1:]
    grad_pred_y = pred[:, :, :-1, :] - pred[:, :, 1:, :]

    grad_target_x = target[:, :, :, :-1] - target[:, :, :, 1:]
    grad_target_y = target[:, :, :-1, :] - target[:, :, 1:, :]

    loss_x = F.mse_loss(grad_pred_x, grad_target_x)
    loss_y = F.mse_loss(grad_pred_y, grad_target_y)

    return loss_x + loss_y

class CombinedLoss(torch.nn.Module):
    def __init__(self, alpha=0.8):
        super().__init__()
        self.alpha = alpha  # Weighting factor for MSE vs. gradient loss

    def forward(self, pred, target):
        mse_loss = F.mse_loss(pred, target)
        grad_loss = gradient_loss(pred, target)
        return self.alpha * mse_loss + (1 - self.alpha) * grad_loss

def fourier_loss(pred, target):
    """Computes L1 loss in Fourier space to preserve high-frequency details."""
    pred_fft = torch.fft.fft2(pred)
    target_fft = torch.fft.fft2(target)
    return F.l1_loss(torch.abs(pred_fft), torch.abs(target_fft))

class CombinedFourierLoss(torch.nn.Module):
    def __init__(self, alpha=0.7):
        super().__init__()
        self.alpha = alpha  # Weight for MSE vs. Fourier loss

    def forward(self, pred, target):
        mse_loss = F.mse_loss(pred, target)
        f_loss = fourier_loss(pred, target)
        return self.alpha * mse_loss + (1 - self.alpha) * f_loss

if __name__ == "__main__":
    N = 200000
    res = 256
    batch_sz = 16
    num_epochs = 100
    lr = 0.0001
    num_features = 64

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = UNet(in_channels=1, out_channels=1, init_features=num_features)
    #model = UNet_2(input_channels=1, output_channels=1, base_channels=num_features)
    #model = torch.nn.DataParallel(model)

    model = model.to(device)
    dataset = HDF5Dataset(f"./train_data/unet_{N}_{res}.hdf5")
    #dataset = FH5Dataset("./train_data/farfield_full.h5", "./train_data/phase_masks_full.h5")
    print(f"Training with Parameters: \n Batch Size: {batch_sz} \n Num_Epochs: {num_epochs} \n "
         f"lr: {lr} \n num_features: {num_features} \n dataset: {dataset.file_path}")

    dataset_size = len(dataset)
    train_size = int(0.8 * dataset_size)  # 80% training, 20% validation

    val_size = dataset_size - train_size

    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_dataloader = DataLoader(train_dataset, batch_size=batch_sz, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_sz, shuffle=True)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    #loss_fc = CombinedLoss(alpha=0.8)
    #loss_fc = CombinedFourierLoss(alpha=0.7)
    loss_fc = torch.nn.MSELoss()
    #loss_fc = torch.nn.HuberLoss()
    # loss_fc = torch.nn.L1Loss()
    train_unet(model, train_dataloader, val_dataloader, optimizer, loss_fc, num_epochs=num_epochs)

    print("Training finished")
