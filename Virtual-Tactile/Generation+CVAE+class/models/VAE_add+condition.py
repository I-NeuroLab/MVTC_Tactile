import torch
import torch.nn as nn
import torch.nn.functional as F

class VAE(nn.Module):
    def __init__(self, input_dim=7500, latent_dim=1000, device='cuda', num_classes = 6, num_material_classes=10, unique_pairs_map=None):
        super(VAE, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.device = device
        
        self.half_dim = int(latent_dim / 2) 
        
        self.num_classes = num_classes
        self.num_material_classes = num_material_classes
        self.total_cond_dim = num_classes + num_material_classes
        
        self.means_percep = nn.Parameter(torch.randn(self.num_classes, self.half_dim))       # (6, 500)
        self.means_mat = nn.Parameter(torch.randn(self.num_material_classes, self.half_dim)) # (10, 500)
        
        # encoder
        self.encoder = nn.Sequential(
            nn.Conv1d(1 + self.total_cond_dim, 16, kernel_size=50, stride=2, padding=1),
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
        self.reverse_layer = nn.Linear(self.latent_dim + self.total_cond_dim, self.flatten_dim)
        
        # decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(64, 32, kernel_size=50, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(32, 16, kernel_size=50, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(16, 1, kernel_size=50, stride=2, padding=1)
            )
        
        # Classifiers (각각 자기 영역만 보고 맞추도록 함)
        # z_p(앞 500) -> Perception 예측
        self.classifier_percep = nn.Sequential(
            nn.Linear(self.half_dim, 128),
            nn.ReLU(),
            nn.Linear(128, self.num_classes)
        )
        # z_m(뒤 500) -> Material 예측
        self.classifier_mat = nn.Sequential(
            nn.Linear(self.half_dim, 128),
            nn.ReLU(),
            nn.Linear(128, self.num_material_classes)
        )
        
    def compute_split_KL(self, mean, log_var, y_percep, y_mat):
        """
        [핵심] KL Divergence를 절반씩 쪼개서 계산
        - 앞 500개는 Perception Mean과 비교
        - 뒤 500개는 Material Mean과 비교
        """
        mean_p, mean_m = self.split_latent(mean)
        logvar_p, logvar_m = self.split_latent(log_var)
        
        target_mean_p = self.means_percep[y_percep]
        target_mean_m = self.means_mat[y_mat]
        
        # KL_percep
        kl_p = -0.5 * torch.sum(1 + logvar_p - (mean_p - target_mean_p).pow(2) - logvar_p.exp(), dim=1)
        # KL_material
        kl_m = -0.5 * torch.sum(1 + logvar_m - (mean_m - target_mean_m).pow(2) - logvar_m.exp(), dim=1)
        
        return (kl_p + kl_m).mean()

    def centroid_loss(self, z, y_percep, y_mat):
        """
        [핵심] Centroid Loss도 절반씩 쪼개서 계산
        """
        z_p, z_m = self.split_latent(z)
        
        target_p = self.means_percep[y_percep]
        target_m = self.means_mat[y_mat]
        
        loss_p = ((z_p - target_p)**2).mean()
        loss_m = ((z_m - target_m)**2).mean()
        
        return loss_p + loss_m
    
    def split_latent(self, z):
        # z를 (Batch, 1000) -> (Batch, 500), (Batch, 500)으로 쪼갬
        z_p = z[:, :self.half_dim]
        z_m = z[:, self.half_dim:]
        return z_p, z_m

    def get_conv_output_len(self, input_len):
        def conv1d_out(l_in, k=50, s=2, p=1):
            return (l_in + 2 * p - (k - 1) - 1) // s + 1
    
        l1 = conv1d_out(input_len)
        l2 = conv1d_out(l1)
        l3 = conv1d_out(l2)
        return l3  # final length after encoder
        
    def encode(self, x, y_combined):
        # x: (batch, 1, 7500)
        # y_onehot: (batch, num_classes)
        y_expand = y_combined.unsqueeze(-1).expand(-1, -1, x.shape[-1])
        xy = torch.cat([x, y_expand], dim=1)
        
        h = self.encoder(xy)
        h = h.view(h.size(0), -1)
        mean, logvar = self.mean_layer(h), self.logvar_layer(h)
        return mean, logvar
    
    def reparameterization(self, mean, logvar):
        std = torch.exp(0.5 * logvar) + 1e-6
        epsilon = torch.randn_like(std).to(self.device)
        return mean + std * epsilon

    def decode(self, x, y_combined):
        xy = torch.cat([x, y_combined], dim=1)
        h = self.reverse_layer(xy)
        h = h.view(h.size(0), 64, self.conv_out_len)
        x_hat = self.decoder(h)
        
        if x_hat.shape[-1] != self.input_dim:
            diff = self.input_dim - x_hat.shape[-1]
            x_hat = F.pad(x_hat, (0, diff)) if diff > 0 else x_hat[:, :, :self.input_dim]
        return x_hat

    def forward(self, x, y_percep, y_mat):
        if y_percep.dim() == 2: y_percep = y_percep.squeeze(1)
        if y_mat.dim() == 2: y_mat = y_mat.squeeze(1)
            
        # One-hot Encoding
        y_onehot = F.one_hot(y_percep.long(), num_classes=self.num_classes).float()
        y_onehot_mat = F.one_hot(y_mat.long(), num_classes=self.num_material_classes).float()
        y_combined = torch.cat([y_onehot, y_onehot_mat], dim=1)
        
        # Encoding (Condition 포함!)
        mean, logvar = self.encode(x, y_combined)
        z = self.reparameterization(mean, logvar)
        x_hat = self.decode(z, y_combined)
        
        # [핵심] Classifier도 쪼개서 넣음
        z_p, z_m = self.split_latent(z)
        out_cls_percep = self.classifier_percep(z_p) # 앞쪽 z는 Perception만 맞춤
        out_cls_mat = self.classifier_mat(z_m)       # 뒤쪽 z는 Material만 맞춤
        
        return x_hat, mean, logvar, out_cls_percep, out_cls_mat, z

