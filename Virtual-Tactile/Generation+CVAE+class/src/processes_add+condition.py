import torch
import torch.nn as nn
import numpy
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, balanced_accuracy_score
from collections import Counter
import torchaudio

devices = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import numpy

def loss_function(x, x_hat, mean, log_var, output1, output2, y_percep, y_mat, z, model, alpha=1.0, beta_centroid=1.0, gamma_ortho= 10.0):
    reproduction_loss = nn.functional.mse_loss(x_hat, x, reduction='mean')
    KLD = model.compute_split_KL(mean, log_var, y_percep, y_mat)
    class_loss1 = nn.functional.cross_entropy(output1, y_percep)
    class_loss2 = nn.functional.cross_entropy(output2, y_mat)
    centroid_loss = model.centroid_loss(z, y_percep, y_mat)
    
    total_loss = reproduction_loss + KLD + alpha * class_loss1 + alpha * class_loss2 + beta_centroid * centroid_loss
    return total_loss

def train_one_epoch_VAE(model, dataloader, optimizer, beta=1.0):
    model.train()
    total_loss = 0.0
    n_samples = 0
    total_corr_percep = 0.0
    total_corr_mat = 0.0

    for x, y_percep, y_mat in dataloader:
        x = x.to(devices)
        y_percep = y_percep.to(devices).long()
        y_mat = y_mat.to(devices).long()
        
        optimizer.zero_grad()
        x_hat, mean, log_var, output1, output2, z = model(x, y_percep, y_mat)
        loss = loss_function(x, x_hat, mean, log_var, output1, output2, y_percep, y_mat, z, model, alpha=1.0, beta_centroid=1.0)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        n_samples += x.size(0)
        
        pred_p = torch.argmax(output1, dim=1)
        pred_m = torch.argmax(output2, dim=1)
        
        total_corr_percep += (pred_p == y_percep).sum().item()
        total_corr_mat += (pred_m == y_mat).sum().item()
        
    return total_loss / n_samples, total_corr_percep / n_samples, total_corr_mat / n_samples

@torch.no_grad()
def validation_VAE(model, dataloader):
    model.eval()
    total_loss = 0.0
    total_corr_percep = 0.0
    total_corr_mat = 0.0
    n_samples = 0

    for x, y_percep, y_mat in dataloader:
        x = x.to(devices)
        y_percep = y_percep.to(devices).long()
        y_mat = y_mat.to(devices).long()
        
        x_hat, mean, log_var, output1, output2, z = model(x, y_percep, y_mat)
        loss = loss_function(x, x_hat, mean, log_var, output1, output2, y_percep, y_mat, z, model, alpha=1.0, beta_centroid=1.0)
        total_loss += loss.item()
        n_samples += x.size(0)
        
        pred_p = torch.argmax(output1, dim=1)
        pred_m = torch.argmax(output2, dim=1)
        
        total_corr_percep += (pred_p == y_percep).sum().item()
        total_corr_mat += (pred_m == y_mat).sum().item()
        
    return total_loss / n_samples, total_corr_percep / n_samples, total_corr_mat / n_samples

@torch.no_grad()
def extract_latent(model, dataloader):
    model.eval()
    all_latent = []
    all_y = []
    all_x = []
    all_m = []
    all_recon = []
    for x, y, m in dataloader:
        x = x.to(devices)
        y = y.to(devices).long()
        m = y.to(devices).long()
        if y.dim() == 2 and y.shape[1] == 1:
            y = y.squeeze(1)
        if m.dim() == 2 and m.shape[1] == 1:
            m = m.squeeze(1)
            
        y_onehot = nn.functional.one_hot(y, num_classes=6).float()
        y_onehot_mat = nn.functional.one_hot(m, num_classes=10).float()
        y_combined = torch.cat([y_onehot, y_onehot_mat], dim=1)
    
        mean, logvar = model.encode(x, y_combined)
        x_recon, _, _, _, _, _ = model(x, y, m)
        z = model.reparameterization(mean, logvar)
        all_latent.append(z.cpu())
        all_m.append(m.cpu())
        all_y.append(y.cpu())
        all_x.append(x.cpu())
        all_recon.append(x_recon.cpu())
    return torch.cat(all_latent, dim=0), torch.cat(all_m, dim=0), torch.cat(all_y, dim=0), torch.cat(all_x, dim=0), torch.cat(all_recon, dim=0)

@torch.no_grad()
def extract_latent_valid(model, num_samples_per_class=10, num_classes=6, latent_dim=1000, num_mat_classes=10):
    
    model.eval()
    all_latent = []
    all_y = []
    all_m = []
    all_recon = []
    half_dim = model.half_dim
    
    if hasattr(model, 'num_classes'):
        num_classes = model.num_classes
    if hasattr(model, 'num_mat_classes'):
        num_mat_classes = model.num_mat_classes
    else:
        num_classes = 6 # Default fallback
        num_mat_classes = 10 # Default fallback
        
    for class_idx in range(num_classes):
        for mat_idx in range(num_mat_classes):
            # Create labels for this class
            
            y_percep = torch.full((num_samples_per_class,), class_idx, dtype=torch.long).to(devices)
            y_mat = torch.full((num_samples_per_class,), mat_idx, dtype=torch.long).to(devices)
            
            y_percep_oh = F.one_hot(y_percep, num_classes=model.num_classes).float()
            y_mat_oh = F.one_hot(y_mat, num_classes=model.num_material_classes).float()
            y_combined = torch.cat([y_percep_oh, y_mat_oh], dim=1)
            
            # 2. [핵심] 레고 조립 (Concatenation)
            # Perception 창고에서 해당 클래스의 평균을 가져옴
            mu_p = model.means_percep[y_percep] # (N, 500)
            
            # Material 창고에서 해당 소재의 평균을 가져옴
            mu_m = model.means_mat[y_mat]       # (N, 500)
            
            # 노이즈 추가
            z_noise = torch.randn(num_samples_per_class, latent_dim).to(devices)
            z_p_noise, z_m_noise = model.split_latent(z_noise)
            
            # 각각 더하고 이어붙임 (Concat)
            z_p = mu_p + z_p_noise
            z_m = mu_m + z_m_noise
            z_total = torch.cat([z_p, z_m], dim=1) # (N, 1000)
            
            # 3. Decoding
            x_recon = model.decode(z_total, y_combined)
            
            all_latent.append(z_total.cpu())
            all_y.append(y_percep.cpu())
            all_m.append(y_mat.cpu())
            all_recon.append(x_recon.cpu())
    
    return torch.cat(all_latent, dim=0), torch.cat(all_m, dim=0), torch.cat(all_y, dim=0), torch.cat(all_recon, dim=0)
