import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import h5py
import numpy as np
import matplotlib.pyplot as plt
from mlp_model import MLP, HDF5Dataset

B = 1000
model_path = "mlp_model.pth"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

dataset = HDF5Dataset("./data/validation_data_1_spot.hdf5")
dataloader = data.DataLoader(dataset, batch_size=B, shuffle=True)

def load_model(model_path):
    model = MLP().to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

model = load_model(model_path)
data_iter = iter(dataloader)
inputs, targets = next(data_iter)
inputs, targets = inputs.to(device), targets.to(device)
predictions = model(inputs).detach().cpu().numpy()

criterion = nn.MSELoss()
mse_loss = criterion(torch.tensor(predictions), targets.cpu()).item()
print(f"Mean Squared Error: {mse_loss:.6f}")

# T = 5  #
# cols = 2
# rows = T
# fig, axs = plt.subplots(rows, cols, figsize=(10, 5 * rows))
#
# for i in range(T):
#     axs[i, 0].imshow(targets[i].cpu().numpy(), cmap='gray')
#     axs[i, 0].set_title(f"Actual Mask {i+1}")
#     axs[i, 1].imshow(predictions[i], cmap='gray')
#     axs[i, 1].set_title(f"Predicted Mask {i+1}")
#
# plt.tight_layout()
# plt.show()