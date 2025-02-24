import torch
import torch.utils.data as data
import torch.nn as nn
import h5py
import numpy as np
import matplotlib.pyplot as plt
from mlp import MLP, HDF5Dataset
from red_tweezers import calculate_phase_mask

B = 2000
model_path = r"./models/mlp_model_5_spot_512-1024-2048-800k.pth"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

dataset = HDF5Dataset("./data/5_spots_discrete_continuous_validation.hdf5")
dataloader = data.DataLoader(dataset, batch_size=B, shuffle=False)

def load_model(model_path):
    model = MLP().to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

model = load_model(model_path)
data_iter = iter(dataloader)
inputs, targets = next(data_iter)

for i in range(5):
    inputs, targets = inputs.to(device), targets.to(device)
    predictions = model(inputs).detach().cpu().numpy()

    criterion = nn.MSELoss()
    mse_loss = criterion(torch.tensor(predictions), targets.cpu()).item()
    print(f"MSE Loss for {i + 1} spots: {mse_loss}")

    inputs, targets = next(data_iter)






