import torch
import torch.nn as nn

class VAE(nn.Module):
    def __init__(self, input_shape, latent_dim, device, y_dim=5):
        super(VAE, self).__init__()
        self.input_shape = input_shape
        self.latent_dim = latent_dim
        self.device = device
        self.y_dim = y_dim
        
        # encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1 + y_dim, 16, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            )
        
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1 + y_dim, *input_shape[1:])
            conv_out = self.encoder(dummy_input)
            self.flatten_dim = conv_out.view(1, -1).shape[1]
            self.decoder_init_shape = conv_out.shape[1:]

        self.mean_layer = nn.Linear(self.flatten_dim, latent_dim)
        self.logvar_layer = nn.Linear(self.flatten_dim, latent_dim)

        self.decoder_input = nn.Linear(latent_dim + y_dim, self.flatten_dim)

        # decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
            )
        
        self.classifier1 = nn.Sequential(
            nn.Linear(self.latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 5)
        )
        
        self.classifier2 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(16 * 16, 128),
            nn.ReLU(),
            nn.Linear(128, 5)
        )
    
    def encode(self, x, y):
        if y.dim() == 1:
            y = y.unsqueeze(-1)
        y_expand = y.view(-1, 1, 1, 1).expand(-1, 1, x.shape[2], x.shape[3])
        x_cat = torch.cat([x, y_expand], dim=1)
        h = self.encoder(x_cat)
        h_flat = h.view(h.size(0), -1)
        mean = self.mean_layer(h_flat)
        logvar = self.logvar_layer(h_flat)
        return mean, logvar

    def reparameterize(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std
    
    def decode(self, z, y):
        if y.dim() == 1:
            y = y.unsqueeze(-1)
        z_cat = torch.cat([z, y], dim=1)
        h = self.decoder_input(z_cat).view(-1, *self.decoder_init_shape)  # 이 부분이 핵심!
        x_recon = self.decoder(h)
        target_size = self.input_shape[1:]
        if x_recon.shape[2:] != target_size:
            x_recon = torch.nn.functional.interpolate(x_recon, size=target_size, mode='bilinear', align_corners=False)
        return x_recon

    def c_classification(self, x_recon):
        return self.classifier2(x_recon)

    def forward(self, x, y):
        mean, logvar = self.encode(x, y)
        z = self.reparameterize(mean, logvar)
        x_recon = self.decode(z, y)
        output1 = self.classifier1(z)
        output2 = self.c_classification(x_recon)
        
        return x_recon, mean, logvar, output1, output2