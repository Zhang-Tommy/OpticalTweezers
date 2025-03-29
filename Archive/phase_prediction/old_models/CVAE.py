import torch
import torch.nn as nn
import torch.utils.data as data
import jax
import h5py

def repeat_n(ten, n):
    return torch.stack([ten] * n, dim=0)

def beta_schedule(i):
    return jax.nn.sigmoid(20 * (i - 0.5)).item()

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

# class MLPEncoder(torch.nn.Module):
#     def __init__(self, input_dim, output_dim, mlp_hidden_dim=32):
#         super(MLPEncoder, self).__init__()
#         self.input_dim = input_dim
#         self.output_dim = output_dim
#
#         model = nn.Sequential()
#         model.add_module("dense1", nn.Linear(input_dim, mlp_hidden_dim))
#         model.add_module("act1", nn.ReLU())
#         model.add_module("dense2", nn.Linear(mlp_hidden_dim, output_dim))
#         self.model = model
#         ############################
#
#     def forward(self, x):
#         return self.model(x)
#
# class MLPDecoder(torch.nn.Module):
#     def __init__(self, input_dim, output_dim, mlp_hidden_dim=32):
#         super(MLPDecoder, self).__init__()
#         self.input_dim = input_dim
#         self.output_dim = output_dim
#
#         model = nn.Sequential()
#         model.add_module("dense1", nn.Linear(input_dim, mlp_hidden_dim))
#         model.add_module("act1", nn.ReLU())
#         model.add_module("dense2", nn.Linear(mlp_hidden_dim, output_dim))
#         self.model = model
#         ############################
#
#     def forward(self, x):
#         return self.model(x)

# Output is output_dim


# Input_dim is parameters_dim
class CNNEncoder(torch.nn.Module):
    def __init__(self, input_dim, output_dim, image_size=512*512):
        super(CNNEncoder, self).__init__()

        self.encoder_cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1),  # Output: (batch_size, 32, 256, 256)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),  # Output: (batch_size, 64, 128, 128)
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=4, stride=2, padding=1),  # Output: (batch_size, 64, 64, 64)
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=4, stride=2, padding=1),  # Output: (batch_size, 64, 32, 32)
            nn.ReLU(),
        )
        self.encoder_fc = nn.Sequential(
            nn.Linear(64 * (image_size // 16) * (image_size // 16) + input_dim, output_dim),
            nn.ReLU(),
        )

    def forward(self, x, y): # x is parameters, y is known reference phase mask
        # feed in y through CNN
        cnn_out = self.encoder_cnn(y)
        cnn_out = cnn_out.flatten()

        fc_out = self.encoder_fc(torch.cat([cnn_out, x], dim=1)) # combine parameters and flattened phase mask features

        # output is going to be output_dim
        return self.model(fc_out)

# Input is parameters_dimension, output is output_dim
class MLPEncoder(torch.nn.Module):
    def __init__(self, input_dim, output_dim, image_size=512 * 512):
        super(MLPEncoder, self).__init__()

        self.encoder_fc = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.encoder_fc(x)

# input is parameter_dimension + latent_var_dimension, output is 512 x 512
class CNNDecoder(torch.nn.Module):
    def __init__(self, input_dim, latent_dim, image_size=512, cnn_channels=64, mlp_hidden_dim=256):
        super(CNNDecoder, self).__init__()

        self.decoder_fc = nn.Sequential(
            nn.Linear(input_dim + latent_dim, mlp_hidden_dim),  # (batch_size, 256)
            nn.ReLU(),
            nn.Linear(mlp_hidden_dim, cnn_channels * (image_size // 4) * (image_size // 4)),
            nn.ReLU(),
        ) # output is (batch_size, 64 * 128 * 128)

        self.decoder_cnn = nn.Sequential(
            nn.ConvTranspose2d(cnn_channels, 32, kernel_size=4, stride=2, padding=1),  # (batch_size, 32, 256, 256)
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1),  # (batch_size, 1, 512, 512)
            nn.Tanh(),  # Output in [-1, 1]
        )

    def forward(self, x, z):
        # given input parameters and sampled latent variable, output 512 x 512 phase mask
        fc_out = self.decoder_fc(x + z)
        # Reshape: (batch_size, 64, 128, 128)
        cnn_out = self.decoder_cnn(fc_out.view(fc_out.size(0), 64, self.image_size // 4, self.image_size // 4))

        return cnn_out.squeeze(1) # batch_size x 512 x 512

class ContinuousCVAE(torch.nn.Module):
    def __init__(self, latent_dim, importance, decoder, prior):
        '''
        latent_dim: dimension of the continuous latent space
        importance: network to encode the importance weight
        decoder: network to decode the output
        prior: network to encode the prior
        '''

        super(ContinuousCVAE, self).__init__()
        self.latent_dim = latent_dim
        self.importance = importance  # q
        self.decoder = decoder  # p(phi)
        self.prior = prior  # p(theta)

        self.mean_projection_encoder = nn.Linear(self.importance.output_dim, self.latent_dim)
        self.logvar_projection_encoder = nn.Linear(self.importance.output_dim, self.latent_dim)

        self.mean_projection_decoder = nn.Linear(self.decoder.output_dim, self.decoder.output_dim)
        self.logvar_projection_decoder = nn.Linear(self.decoder.output_dim, self.decoder.output_dim)

    def encode_importance(self, x, y):
        '''Computes mean and log(covariance) of q(z|x,y), assumes normal distribution'''
        xy = torch.cat([x, y], dim=-1)
        z_mu = self.mean_projection_encoder(self.importance(xy))
        z_logvar = self.logvar_projection_encoder(self.importance(xy))

        ############################

        return z_mu, z_logvar

    def encode_prior(self, x):
        '''Computes mean and log(covariance) of p(z|x), assumes normal distribution'''

        z_mu = self.mean_projection_encoder(self.prior(x))
        z_logvar = self.logvar_projection_encoder(self.prior(x))
        ############################

        return z_mu, z_logvar

    def reparameterize(self, mu, logvar, n=1):
        '''samples from a normal distributions parameterized by mu and logvar. Uses PyTorch built-in reparameratization trick'''
        prob = torch.distributions.MultivariateNormal(loc=mu, covariance_matrix=torch.diag_embed(torch.exp(logvar)))
        return prob.rsample((n,))

    def decode(self, x, z):
        '''Computes mean and log(covariance) of p(y|x,z), assumes normal distribution'''

        xz = torch.cat([x, z], dim=-1)
        y_mu = self.mean_projection_decoder(self.decoder(xz))
        y_logvar = self.logvar_projection_decoder(self.decoder(xz))
        ############################
        return y_mu, y_logvar

    def forward(self, x, y, n=1):
        '''forward pass of the cvae model'''

        #  get p(z|x,(y))
        if self.training:
            z_mu, z_logvar = self.encode_importance(x, y)
        else:
            z_mu, z_logvar = self.encode_prior(x)
        # sample from p(z|x,(y)) n times
        z = self.reparameterize(z_mu, z_logvar, n)
        # get p(y|x,z)
        y_mu, y_logvar = self.decode(repeat_n(x, n), z)

        return z_mu, z_logvar, y_mu, y_logvar

    def sample(self, x, num_samples=8, num_latent_samples=8):
        '''samples from p(y|x,z) where z~p(z|x). Need to specify the number z and y samples to draw'''
        _, _, y_mu, y_logvar = self.forward(x, None, num_latent_samples)
        return self.reparameterize(y_mu, y_logvar, num_samples)

    def elbo(self, x, y, z_samples=1, beta=1.):
        '''Compute ELBO for CVAE with continuous latent space. Optional: beta term that weigh kl divergence term'''

        q_mu, q_logvar, y_mu, y_logvar = self(x, y,
                                              z_samples)  # get parameters for q(z|x,y) and p(y|x,z) where z~q(z|x,y)
        p_mu, p_logvar = self.encode_prior(x)  # get parameters for p(z|x)

        # construct the distributions
        y_prob = torch.distributions.MultivariateNormal(loc=y_mu, covariance_matrix=torch.diag_embed(
            torch.exp(y_logvar)))  # p(y|x, z)
        q = torch.distributions.MultivariateNormal(loc=q_mu,
                                                   covariance_matrix=torch.diag_embed(torch.exp(q_logvar)))  # q(z|x,y)
        p = torch.distributions.MultivariateNormal(loc=p_mu,
                                                   covariance_matrix=torch.diag_embed(torch.exp(p_logvar)))  # p(z|x)

        loglikelihood = -y_prob.log_prob(repeat_n(y, z_samples)).mean()  # log likelihood of data
        kl_div = torch.distributions.kl.kl_divergence(q, p).mean()  # q_z * (log(q_z) - log(p_z))

        return loglikelihood + beta * kl_div


def main():
    batch_size = 1
    dataset = HDF5Dataset("./data/5_spots_discrete_continuous_validation.hdf5")
    train_dataloader = data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    latent_dim = 8  # size of latent space
    # Given history, predict future
    # Given inputs, predict outputs
    # Given parameters, predict 512 x 512 phase mask
    input_param_size = 16  # 5x4x4 spot data

    mlp_hidden_dim = 8
    enc_out_dim = 64

    encoder = CNNEncoder(input_param_size, enc_out_dim)
    prior = MLPEncoder(input_param_size, enc_out_dim)

    decoder = CNNDecoder(input_param_size, latent_dim)

    cvae = ContinuousCVAE(latent_dim, encoder, decoder, prior)

    num_epochs = 3
    learning_rate = 0.001
    optimizer = torch.optim.Adam(cvae.parameters(), lr=learning_rate)

    # train model
    z_samples = 16
    cvae.train()

    for epoch in range(num_epochs):
        running_loss = 0
        beta = beta_schedule((epoch + 1) / num_epochs)  # we slowly increase the weighting on the KL divergence, following https://openreview.net/forum?id=Sy2fzU9gl
        for batch_idx, (history, future) in enumerate(train_dataloader):
            q_mu, q_logvar, y_mu, y_logvar = cvae(history, future)
            p_mu, p_logvar = cvae.encode_prior(history)
            optimizer.zero_grad()
            loss = cvae.elbo(history, future, z_samples, beta)
            loss.backward()
            running_loss += loss.detach().cpu().numpy()
            optimizer.step()

            if batch_idx % 100 == 0:
                print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{batch_idx}/{len(train_dataloader)}], Loss: {loss.item():.4f}')

        print(f'======= Epoch {epoch + 1} completed with average loss: {running_loss / len(train_dataloader):.4f} =======')

    print("Training finished!")

if __name__ == '__main__':
    main()