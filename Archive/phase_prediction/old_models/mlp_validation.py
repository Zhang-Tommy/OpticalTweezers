import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import h5py
import numpy as np
import matplotlib.pyplot as plt
from mlp import MLP, HDF5Dataset
from red_tweezers import calculate_phase_mask
B = 1000
model_path = "models/mlp_model_300k_fixed.pth"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

dataset = HDF5Dataset("./data/10_spots_validation.hdf5")
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

# criterion = nn.MSELoss()
# mse_loss = criterion(torch.tensor(predictions), targets.cpu()).item()
#print(f"Mean Squared Error: {mse_loss:.6f}")
""""""
# import pandas as pd
# df_actual = pd.DataFrame(targets.cpu()[0])
# df_predicted = pd.DataFrame(predictions[0])
#
# df_actual.to_csv('actual_mask.csv', index=False, header=False)
# df_predicted.to_csv('predicted_mask.csv', index=False, header=False)

spots = np.array([

        [[50, -45.210987, -8.000000, 0.000000],
         [1.000000, 0.000000, 0.000000, 0.000000],
         [0.000000, 0.000000, 1.000000, 0.000000],
         [0.000000, 0.000000, 0.000000, 0.000000]],

        [[0, 0.000000, 0.000000, 0.000000],
         [0.000000, 0.000000, 0.000000, 0.000000],
         [0.000000, 0.000000, 0.000000, 0.000000],
         [0.000000, 0.000000, 0.000000, 0.000000]],

        [[0, 0.000000, 0.000000, 0.000000],
         [0.000000, 0.000000, 0.000000, 0.000000],
         [0.000000, 0.000000, 0.000000, 0.000000],
         [0.000000, 0.000000, 0.000000, 0.000000]],

        [[0, 0.000000, 0.000000, 0.000000],
         [0.000000, 0.000000, 0.000000, 0.000000],
         [0.000000, 0.000000, 0.000000, 0.000000],
         [0.000000, 0.000000, 0.000000, 0.000000]],

        [[0, 0.000000, 0.000000, 0.000000],
         [0.000000, 0.000000, 0.000000, 0.000000],
         [0.000000, 0.000000, 0.000000, 0.000000],
         [0.000000, 0.000000, 0.000000, 0.000000]],

        [[0, 0.000000, 0.000000, 0.000000],
         [0.000000, 0.000000, 0.000000, 0.000000],
         [0.000000, 0.000000, 0.000000, 0.000000],
         [0.000000, 0.000000, 0.000000, 0.000000]],

        [[0, 0.000000, 0.000000, 0.000000],
         [0.000000, 0.000000, 0.000000, 0.000000],
         [0.000000, 0.000000, 0.000000, 0.000000],
         [0.000000, 0.000000, 0.000000, 0.000000]],

        [[0, 0.000000, 0.000000, 0.000000],
         [0.000000, 0.000000, 0.000000, 0.000000],
         [0.000000, 0.000000, 0.000000, 0.000000],
         [0.000000, 0.000000, 0.000000, 0.000000]],

        [[0, 0.000000, 0.000000, 0.000000],
         [0.000000, 0.000000, 0.000000, 0.000000],
         [0.000000, 0.000000, 0.000000, 0.000000],
         [0.000000, 0.000000, 0.000000, 0.000000]],

        [[0, 0.000000, 0.000000, 0.000000],
         [0.000000, 0.000000, 0.000000, 0.000000],
         [0.000000, 0.000000, 0.000000, 0.000000],
         [0.000000, 0.000000, 0.000000, 0.000000]]
    ])
spots_tensor = torch.tensor(spots, dtype=torch.float32)
spots_tensor = spots_tensor.unsqueeze(0)
spots_tensor = spots_tensor.to(device)

single_spot_prediction = model(spots_tensor).detach().cpu().numpy()

#print(single_spot_prediction)


predicted_mask = single_spot_prediction[0]  # Remove batch dimension if needed

#clipped_mask = np.clip(predicted_mask, -np.pi, np.pi)

reference_mask = calculate_phase_mask(spots, 10)[0]

#print(clipped_mask - reference_mask)

plt.figure(figsize=(6, 6))
plt.imshow(np.abs(predicted_mask - reference_mask), cmap='gray')  # Use 'gray' colormap for better visualization
plt.colorbar(label="Phase Value")
plt.title("Predicted Mask")
plt.axis("off")  # Remove axis for a cleaner look
plt.show()

# """"""
# T = 5  # Number of samples to display
# cols = 3  # Actual, Predicted, and Difference
# rows = T
# fig, axs = plt.subplots(rows, cols, figsize=(10, 2 * rows), constrained_layout=True)
#
# # Set column titles
# axs[0, 0].set_title("Actual Mask", fontsize=12)
# axs[0, 1].set_title("Predicted Mask", fontsize=12)
# axs[0, 2].set_title("Difference (Abs Error)", fontsize=12)
#
# for i in range(T):
#     actual = targets[i].cpu().numpy()
#     predicted = predictions[i]
#     difference = np.abs(actual - predicted)  # Absolute error
#
#     axs[i, 0].imshow(actual, cmap='gray')
#     axs[i, 1].imshow(predicted, cmap='gray')
#     axs[i, 2].imshow(difference, cmap='gray')  # Use 'hot' colormap to highlight differences
#
#     # Remove axis labels for cleaner visualization
#     for j in range(cols):
#         axs[i, j].axis("off")
#
# plt.show()
