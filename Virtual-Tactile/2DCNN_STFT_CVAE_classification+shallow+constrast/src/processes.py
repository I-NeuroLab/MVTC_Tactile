import torch
import torch.nn as nn
import numpy
import torch.nn.functional as F
from sklearn.metrics import balanced_accuracy_score
from collections import Counter
import torchaudio

devices = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def normalize_2d_tensor(input_tensor):
    mean = input_tensor.mean(dim=(1,2), keepdim=True)
    std = input_tensor.std(dim=(1,2), keepdim=True)

    std = std + 1e-8

    normalized_tensor = (input_tensor - mean) / std
    return normalized_tensor

class SupConLoss(torch.nn.Module):
    def __init__(self, temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        """
        features: (batch, latent_dim)
        labels: (batch,)
        """
        device = features.device
        batch_size = features.shape[0]
        features = F.normalize(features, dim=1)
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)

        # cosine similarity
        anchor_dot_contrast = torch.div(torch.matmul(features, features.T), self.temperature)
        # self-comparison mask
        logits_mask = torch.ones_like(mask).fill_diagonal_(0)
        mask = mask * logits_mask

        exp_logits = torch.exp(anchor_dot_contrast) * logits_mask
        log_prob = anchor_dot_contrast - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-8)
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-8)
        loss = -mean_log_prob_pos.mean()
        return loss
    
def train_one_epoch(model, dataloader, optimizer, alpha=1.0, beta=1.0, gamma=1.0, temperature=0.07):
    """
    alpha: reconstruction+KLD 가중치
    beta: classification loss 가중치
    gamma: SupCon loss 가중치
    """
    model.train()
    supcon_criterion = SupConLoss(temperature=temperature)
    overall_loss = 0.
    total_samples = 0
    correct1 = 0
    correct2 = 0

    for inputs, targets in dataloader:
        inputs = inputs.to(devices)
        targets = targets.to(devices).long()
        batch_size = targets.size(0)

        # 데이터 전처리
        spectrogram = torchaudio.transforms.Spectrogram(
            n_fft=5000,
            win_length=5000,
            hop_length=30,
            power=2.0
        ).to(devices)
        spec = spectrogram(inputs)
        inputs = normalize_2d_tensor(spec[:, 0:300, :]).unsqueeze(1)
        del spec

        optimizer.zero_grad()
        # (x_hat, mean, log_var, z_out, c_out): z_out=(batch, latent_dim), c_out=(batch, num_classes)
        x_hat, mean, log_var, z_out, c_out = model(inputs, targets)

        # SupCon loss
        supcon_loss = supcon_criterion(z_out, targets)

        # classification loss
        class_loss = F.cross_entropy(c_out, targets)

        # reconstruction + KLD
        reconstruction_loss = F.mse_loss(x_hat, inputs, reduction='mean')
        KLD = -0.5 * torch.mean(1 + log_var - mean.pow(2) - log_var.exp())

        # 전체 loss 합산
        total_loss = alpha * (reconstruction_loss + KLD) + beta * class_loss + gamma * supcon_loss
        total_loss.backward()
        optimizer.step()

        overall_loss += total_loss.item()
        total_samples += batch_size

        # accuracy 계산
        preds1 = z_out.argmax(dim=1)
        preds2 = c_out.argmax(dim=1)
        correct1 += (preds1 == targets).sum().item()
        correct2 += (preds2 == targets).sum().item()

    _, _, _, tacc1, tacc2, tacc3 = predict_conditional_vae(
        model, dataloader)

    avg_loss = overall_loss / len(dataloader)
    avg_acc1 = correct1 / total_samples
    avg_acc2 = correct2 / total_samples
    return avg_loss, avg_acc1, avg_acc2, tacc1, tacc2, tacc3


@torch.no_grad()
def validation(model, dataloader, alpha=1.0, beta=1.0, gamma=1.0, temperature=0.07):
    model.eval()
    supcon_criterion = SupConLoss(temperature=temperature)
    overall_loss = 0.
    total_samples = 0
    correct1 = 0
    correct2 = 0

    for inputs, targets in dataloader:
        inputs = inputs.to(devices)
        targets = targets.to(devices).long()
        batch_size = targets.size(0)

        # 데이터 전처리
        spectrogram = torchaudio.transforms.Spectrogram(
            n_fft=5000,
            win_length=5000,
            hop_length=30,
            power=2.0
        ).to(devices)
        spec = spectrogram(inputs)
        inputs = normalize_2d_tensor(spec[:, 0:300, :]).unsqueeze(1)
        del spec

        # (x_hat, mean, log_var, z_out, c_out): z_out=(batch, latent_dim), c_out=(batch, num_classes)
        x_hat, mean, log_var, z_out, c_out = model(inputs, targets)

        # SupCon loss
        supcon_loss = supcon_criterion(z_out, targets)

        # classification loss
        class_loss = F.cross_entropy(c_out, targets)

        # reconstruction + KLD
        reconstruction_loss = F.mse_loss(x_hat, inputs, reduction='mean')
        KLD = -0.5 * torch.mean(1 + log_var - mean.pow(2) - log_var.exp())

        # 전체 loss 합산
        total_loss = alpha * (reconstruction_loss + KLD) + beta * class_loss + gamma * supcon_loss

        overall_loss += total_loss.item()
        total_samples += batch_size

        # accuracy 계산
        preds1 = z_out.argmax(dim=1)
        preds2 = c_out.argmax(dim=1)
        correct1 += (preds1 == targets).sum().item()
        correct2 += (preds2 == targets).sum().item()
        
    _, _, _, tacc1, tacc2, tacc3 = predict_conditional_vae(
        model, dataloader)

    avg_loss = overall_loss / len(dataloader)
    avg_acc1 = correct1 / total_samples
    avg_acc2 = correct2 / total_samples
    return avg_loss, avg_acc1, avg_acc2, tacc1, tacc2, tacc3

@torch.no_grad()
def predict_conditional_vae(model, dataloader, num_classes=5):
    model.eval()
    all_preds1 = []
    all_preds2 = []
    all_preds3 = []
    all_gts = []

    for inputs, targets in dataloader:
        inputs = inputs.to(devices)
        targets = targets.to(devices)

        batch_size = inputs.shape[0]
        # Preprocess: Spectrogram, normalization 등 기존과 동일
        spectrogram = torchaudio.transforms.Spectrogram(
            n_fft=5000,
            win_length=5000,
            hop_length=30,
            power=2.0
        ).to(devices)
        spec = spectrogram(inputs)
        inputs_proc = normalize_2d_tensor(spec[:, 0:300, :]).unsqueeze(1)
        del spec
        
        logits1_all = []
        logits2_all = []
        for c in range(num_classes):
            # One-hot condition
            y_cond = torch.full((batch_size,), c, dtype=torch.long, device=devices)
            #y_onehot = F.one_hot(y_cond, num_classes=num_classes).float()
            x_hat, mean, log_var, logits1, logits2 = model(inputs_proc, y_cond)
            logits1_all.append(logits1.unsqueeze(2))  # (batch, 5, 1)
            logits2_all.append(logits2.unsqueeze(2))
            
            # batch x 5 class prob x c
        # stack: (batch, 5, 5) (각 candidate에 대한 output 5개)
        logits1_all = torch.cat(logits1_all, dim=2)  # (batch, 5, 5)
        logits2_all = torch.cat(logits2_all, dim=2)

        # 각 샘플별로 softmax 적용 후, 가장 높은 candidate index 선택
        probs1 = torch.softmax(logits1_all, dim=1)  # (batch, 5, 5)
        probs2 = torch.softmax(logits2_all, dim=1)
        
        temp1 = torch.sum(logits1_all, dim=2)
        temp2 = torch.sum(logits2_all, dim=2)
        temp3 = temp1+temp2
        
        pred1 = temp1.argmax(dim=1)  # (batch,)
        pred2 = temp2.argmax(dim=1)  # (batch,)
        pred3 = temp3.argmax(dim=1)  # (batch,)
        
        all_preds1.append(pred1.cpu())
        all_preds2.append(pred2.cpu())
        all_preds3.append(pred3.cpu())
        all_gts.append(targets.cpu())

    all_preds1 = torch.cat(all_preds1)
    all_preds2 = torch.cat(all_preds2)
    all_preds3 = torch.cat(all_preds3)
    all_gts = torch.cat(all_gts)

    acc1 = (all_preds1 == all_gts).float().mean()
    acc2 = (all_preds2 == all_gts).float().mean()
    acc3 = (all_preds3 == all_gts).float().mean()
    return all_preds1, all_preds2, all_gts, acc1, acc2, acc3