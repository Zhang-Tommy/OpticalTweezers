import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import h5py
import numpy as np
import matplotlib.pyplot as plt
from mlp import MLP, HDF5Dataset
import cv2 as cv

B = 1000
model_path = "models/mlp_model_600k.pth"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

dataset = HDF5Dataset("./data/10_spots_validation.hdf5")
dataloader = data.DataLoader(dataset, batch_size=B, shuffle=True)

def load_model(model_path):
    model = MLP().to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

model = load_model(model_path)
# data_iter = iter(dataloader)
# inputs, targets = next(data_iter)
# inputs, targets = inputs.to(device), targets.to(device)
# predictions = model(inputs).detach().cpu().numpy()

spots = np.array([

        [[50, -45.210987, 0.000000, 5.800000],
         [1.100000, 0.000000, 0.000000, 0.000000],
         [0.000000, 0.000000, 1.150000, 0.000000],
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
predicted_mask = single_spot_prediction[0]
predicted_mask = (predicted_mask - predicted_mask.min()) / (predicted_mask.max() - predicted_mask.min()) * 255
predicted_mask = predicted_mask.astype(np.uint8)

cv.namedWindow("Mask", cv.WND_PROP_FULLSCREEN)
cv.setWindowProperty("Mask", cv.WND_PROP_FULLSCREEN, cv.WINDOW_FULLSCREEN)
cv.moveWindow("Mask", 0, 0)
cv.resizeWindow("Mask", 512, 512)
cv.imshow("Mask", predicted_mask)

k = cv.waitKey(0)