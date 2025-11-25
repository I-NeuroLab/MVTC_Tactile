import torch
import torch.nn as nn

cuda = torch.cuda.is_available()
device = torch.device('cuda:0' if cuda else 'cpu')
class VAE(nn.Module):
    def __init__(self, input_dim=7500, latent_dim=1000, device=device, num_classes = 5):
        super(VAE, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.centroids = nn.Parameter(torch.randn(self.num_classes, self.latent_dim))

        # encoder
        self.encoder = nn.Sequential(
            nn.Conv1d(1 + num_classes, 16, kernel_size=50, stride=2, padding=1),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Conv1d(16, 32, kernel_size=50, stride=2, padding=1),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Conv1d(32, 64, kernel_size=50, stride=2, padding=1),
            nn.ReLU(),
            )
        
        self.conv_out_len = self.get_conv_output_len(self.input_dim)
        self.flatten_dim = 64 * self.conv_out_len
        
        # latent mean and variance 
        self.mean_layer = nn.Linear(self.flatten_dim, self.latent_dim)
        self.logvar_layer = nn.Linear(self.flatten_dim, self.latent_dim)
        self.reverse_layer = nn.Linear(self.latent_dim + self.num_classes, self.flatten_dim)
        
        # decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(64, 32, kernel_size=50, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(32, 16, kernel_size=50, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(16, 1, kernel_size=50, stride=2, padding=1)
            )
        
        self.classifier1 = nn.Sequential(
            nn.Linear(self.latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, self.num_classes)
        )
        
        self.classifier2_1 = nn.Sequential(
            nn.Conv1d(1, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Conv1d(128, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Conv1d(32, 8, kernel_size=4, stride=2, padding=1),
            nn.ReLU()
        )
        
        self.classifier2_2 = nn.Sequential(
            nn.Linear(5000, 128),
            nn.ReLU(),
            nn.Linear(128, self.num_classes)
        )
        
    def get_conv_output_len(self, input_len):
        def conv1d_out(l_in, k=50, s=2, p=1):
            return (l_in + 2 * p - (k - 1) - 1) // s + 1
    

        l1 = conv1d_out(input_len)
        l2 = conv1d_out(l1)
        l3 = conv1d_out(l2)
        return l3  # final length after encoder
        
    def centroid_loss(self, z, y):
        target_centroids = self.centroids[y]
        return ((z - target_centroids)**2).mean()
    
    def encode(self, x, y_onehot):
        # x: (batch, 1, 7500)
        # y_onehot: (batch, num_classes)
        assert x.shape[0] == y_onehot.shape[0], f"Batch size mismatch: x={x.shape}, y_onehot={y_onehot.shape}"
        y_expand = y_onehot.unsqueeze(-1).expand(-1, -1, x.shape[-1])  # (batch, num_classes, 7500)
        assert x.shape[2] == y_expand.shape[2], f"Length mismatch: x={x.shape}, y_expand={y_expand.shape}"
        xy = torch.cat([x, y_expand], dim=1)  # (batch, 1+num_classes, 7500)
        h = self.encoder(xy)
        h = h.view(h.size(0), -1)
        mean, logvar = self.mean_layer(h), self.logvar_layer(h)
        return mean, logvar

    def reparameterization(self, mean, logvar):
        epsilon = torch.randn_like(logvar).to(device)      
        std = torch.exp(0.5 * logvar) + 1e-6
        z = mean + std*epsilon
        return z

    def decode(self, x, y_onehot):
        xy = torch.cat([x, y_onehot], dim=1)
        h = self.reverse_layer(xy)
        h = h.view(h.size(0), 64, self.conv_out_len)
        x_hat = self.decoder(h)
        if x_hat.shape[-1] != self.input_dim:
            pad_len = self.input_dim - x_hat.shape[-1]
            x_hat = nn.functional.pad(x_hat, (0, max(0, pad_len)))[:, :, :self.input_dim]
        return x_hat

    def forward(self, x, y):
        if y.dim() == 2 and y.shape[1] == 1:
            y = y.squeeze(1)
        y_onehot = nn.functional.one_hot(y, num_classes=self.num_classes).float()
        mean, logvar = self.encode(x, y_onehot)
        z = self.reparameterization(mean, logvar)
        
        x_hat = self.decode(z, y_onehot)
        output1 = self.classifier1(z)
        output2 = self.classifier2_1(x_hat)
        output2 = output2.view(output2.size(0),-1)
        output2 = self.classifier2_2(output2)
        
        return x_hat, mean, logvar, output1, output2, z

