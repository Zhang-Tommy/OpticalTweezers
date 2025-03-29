import torch
import torch.nn as nn
import torch.utils.data as data
import h5py


class HDF5Dataset(data.Dataset):
    def __init__(self, file_path):
        self.file_path = file_path
        with h5py.File(self.file_path, 'r') as f:
            self.length = f['inputs'].shape[0]

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        with h5py.File(self.file_path, 'r') as f:
            x = f['inputs'][idx]  # Shape: (5, 4, 4)
            y = f['outputs'][idx]  # Shape: (512, 512)
            #print(torch.tensor(y, dtype=torch.float32).shape)
        return torch.tensor(x, dtype=torch.float32).squeeze(0).flatten(), torch.tensor(y, dtype=torch.float32).unsqueeze(0)

class CVAE(nn.Module):
    def __init__(self, latent_dim, input_dim, image_size):
        super(CVAE, self).__init__()
        self.latent_dim = latent_dim
        self.input_dim = input_dim
        self.image_size = image_size

        # Encoder
        self.encoder_cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
        )
        self.encoder_fc = nn.Sequential(
            nn.Linear(64 * (image_size // 4) * (image_size // 4) + input_dim, 256),
            nn.ReLU(),
        )
        self.fc_mu = nn.Linear(256, latent_dim)
        self.fc_logvar = nn.Linear(256, latent_dim)

        # Decoder
        self.decoder_fc = nn.Sequential(
            nn.Linear(latent_dim + input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 64 * (image_size // 4) * (image_size // 4)),
            nn.ReLU(),
        )
        self.decoder_cnn = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),  # Output in [-1, 1]
        )

    def encode(self, x, c):
        h = self.encoder_cnn(x)
        h = h.view(h.size(0), -1)
        h = torch.cat([h, c], dim=1)
        h = self.encoder_fc(h)
        return self.fc_mu(h), self.fc_logvar(h)

    def decode(self, z, c):
        h = torch.cat([z, c], dim=1)
        h = self.decoder_fc(h)
        h = h.view(h.size(0), 64, self.image_size // 4, self.image_size // 4)
        return self.decoder_cnn(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, c):
        mu, logvar = self.encode(x, c)
        z = self.reparameterize(mu, logvar)
        return self.decode(z, c), mu, logvar

# Loss function
def loss_function(recon_x, x, mu, logvar):
    recon_loss = nn.functional.mse_loss(recon_x, x, reduction='sum')
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kl_loss

# Training loop
def train(model, dataloader, optimizer):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0
        for batch_idx, (c, x) in enumerate(dataloader):
            optimizer.zero_grad()
            recon_x, mu, logvar = model(x, c)
            loss = loss_function(recon_x, x, mu, logvar)
            loss.backward()
            optimizer.step()
            print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{batch_idx}/{len(train_dataloader)}], Loss: {loss.item():.4f}')

num_epochs = 5
batch_size = 5
learning_rate = 0.00001
dataset = HDF5Dataset("./data/1_spot_10000_samples.hdf5")
train_dataloader = data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

model = CVAE(latent_dim=256, input_dim=16, image_size=512)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
train(model, train_dataloader, optimizer)