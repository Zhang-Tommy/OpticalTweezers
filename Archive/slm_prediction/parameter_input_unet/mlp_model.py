import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import h5py
import numpy as np
import matplotlib.pyplot as plt


class MLP(nn.Module):
    def __init__(self, input_size=160, hidden_sizes=[256, 512, 1024, 2048], output_size=512 * 512):
        super(MLP, self).__init__()
        self.flatten = nn.Flatten()

        layers = []
        prev_size = input_size
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(hidden_size))
            prev_size = hidden_size

        layers.append(nn.Linear(prev_size, output_size))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        x = self.flatten(x)
        x = self.model(x)
        return x.view(-1, 512, 512)

class HDF5Dataset(data.Dataset):
    def __init__(self, file_path):
        self.file_path = file_path
        with h5py.File(self.file_path, 'r') as f:
            self.length = f['inputs'].shape[0]

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        with h5py.File(self.file_path, 'r') as f:
            x = f['inputs'][idx]  # Shape: (10, 4, 4)
            y = f['outputs'][idx]  # Shape: (512, 512)
        return torch.tensor(x, dtype=torch.float32).squeeze(0), torch.tensor(y, dtype=torch.float32)

if __name__ == "__main__":
    B = 200
    epochs = 1
    learning_rate = 0.002

    dataset = HDF5Dataset("./data/train_data_10_new.hdf5")
    dataloader = data.DataLoader(dataset, batch_size=B, shuffle=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = MLP().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            if batch_idx % 10 == 0:
                print(f"Epoch [{epoch + 1}/{epochs}], Batch [{batch_idx + 1}/{len(dataloader)}], Loss: {loss.item():.6f}")

    model_path = "mlp_model_new.pth"
    torch.save(model.state_dict(), model_path)

    def load_model(model_path):
        model = MLP().to(device)
        model.load_state_dict(torch.load(model_path))
        model.eval()
        return model

    # Inference on a batch of data
    model = load_model(model_path)
    data_iter = iter(dataloader)
    inputs, targets = next(data_iter)
    inputs, targets = inputs.to(device), targets.to(device)
    predictions = model(inputs).detach().cpu().numpy()

    # Plot actual vs predicted
    fig, axs = plt.subplots(2, 2, figsize=(10, 10))
    axs[0, 0].imshow(targets[0].cpu().numpy(), cmap='viridis')
    axs[0, 0].set_title("Actual Mask")
    axs[0, 1].imshow(predictions[0], cmap='viridis')
    axs[0, 1].set_title("Predicted Mask")
    axs[1, 0].imshow(targets[1].cpu().numpy(), cmap='viridis')
    axs[1, 0].set_title("Actual Mask")
    axs[1, 1].imshow(predictions[1], cmap='viridis')
    axs[1, 1].set_title("Predicted Mask")
    plt.show()

