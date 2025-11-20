# e:\Vscode\IIT HYD\models\mctnet.py
import math
import torch
import torch.nn as nn

class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class ECA(nn.Module):
    def __init__(self, channels, k_size=3):
        super().__init__()
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size // 2), bias=False)
        self.sig = nn.Sigmoid()
        self.channels = channels
    def forward(self, x):
        s = x.mean(dim=2, keepdim=True)
        w = self.conv(s.transpose(1, 2)).transpose(1, 2)
        w = self.sig(w)
        return x * w

class ALPE(nn.Module):
    def __init__(self, d_model, T):
        super().__init__()
        self.pe = SinusoidalPositionalEncoding(d_model, T)
        self.conv = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1)
        self.eca = ECA(d_model, k_size=3)
    def forward(self, x):
        x = self.pe(x)
        y = x.transpose(1, 2)
        y = self.conv(y)
        y = self.eca(y)
        y = y.transpose(1, 2)
        return x + y

class CNNBranch(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.conv1 = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(d_model)
        self.act = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(d_model)
    def forward(self, x):
        y = x.transpose(1, 2)
        r = y
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)
        y = self.conv2(y)
        y = self.bn2(y)
        y = self.act(y + r)
        return y.transpose(1, 2)

class TransformerBranch(nn.Module):
    def __init__(self, d_model, nhead, ff_dim, dropout):
        super().__init__()
        layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=ff_dim, dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=1)
    def forward(self, x, src_key_padding_mask=None):
        return self.encoder(x, src_key_padding_mask=src_key_padding_mask)

class CTFusionBlock(nn.Module):
    def __init__(self, d_model, nhead, ff_dim, dropout, use_alpe=False, T=128):
        super().__init__()
        self.use_alpe = use_alpe
        self.alpe = ALPE(d_model, T) if use_alpe else None
        self.cnn = CNNBranch(d_model)
        self.trans = TransformerBranch(d_model, nhead, ff_dim, dropout)
        self.norm = nn.LayerNorm(d_model)
    def forward(self, x, mask=None):
        if self.use_alpe:
            x = self.alpe(x)
        hc = self.cnn(x)
        ht = self.trans(x, src_key_padding_mask=None)
        h = self.norm(hc + ht)
        return h

class MCTNet(nn.Module):
    def __init__(self, d_in, d_model, num_classes, nhead=8, ff_dim=512, dropout=0.1, T=128):
        super().__init__()
        self.proj = nn.Linear(d_in, d_model)
        self.stage1 = CTFusionBlock(d_model, nhead, ff_dim, dropout, use_alpe=True, T=T)
        self.stage2 = CTFusionBlock(d_model, nhead, ff_dim, dropout, use_alpe=False, T=T)
        self.stage3 = CTFusionBlock(d_model, nhead, ff_dim, dropout, use_alpe=False, T=T)
        self.pool = nn.AdaptiveMaxPool1d(1)
        hidden = d_model // 2
        self.head = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(hidden, num_classes),
        )
    def forward(self, X, mask=None):
        h = self.proj(X)
        h = self.stage1(h, mask)
        h = self.stage2(h, mask)
        h = self.stage3(h, mask)
        z = h.transpose(1, 2)
        z = self.pool(z).squeeze(-1)
        y = self.head(z)
        return y