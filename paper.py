import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.metrics import confusion_matrix, f1_score, cohen_kappa_score
from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import requests
import os
from pathlib import Path
from tqdm import tqdm
import warnings
import math
warnings.filterwarnings('ignore')
from PIL import Image

# ==================== CONFIGURATION ====================
CONFIG = {
    'data_dir': './eurosat_data',
    'batch_size': 32,
    'epochs': 40,
    'learning_rate': 0.001,
    'n_stages': 3,
    'n_heads': 4,
    'kernel_size': 3,
    'n_classes': 10,
    'd_model': 3,  # RGB channels
    'max_per_class': 2500,
    'num_workers': 2,
    'use_amp': True,
    'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    'temporal_seq_length': 36,  # Simulate 36 timesteps like paper
    'missing_rate': 0.2,  # 20% missing data rate
}

def inspect_eurosat_structure(data_dir):
    """Debug function to inspect actual file structure"""
    data_dir = Path(data_dir)
    print("\n=== EUROSAT Structure Inspection ===")
    
    for class_dir in sorted(data_dir.iterdir()):
        if class_dir.is_dir():
            files = list(class_dir.glob("*"))
            print(f"{class_dir.name}: {len(files)} files")
            if files:
                for f in files[:3]:
                    print(f"  - {f.name}")

# ==================== DATA DOWNLOAD ====================
def download_eurosat(data_dir):
    """Download EUROSAT dataset with SSL bypass and multiple mirror support"""
    data_dir = Path(data_dir)
    data_dir.mkdir(exist_ok=True)
    
    # Multiple URLs (mirrors)
    urls = [
        "http://madm.dfki.de/files/sentinel/EuroSAT.zip",
        "https://madm.dfki.de/files/sentinel/EuroSAT.zip",
    ]
    
    zip_path = data_dir / "EuroSAT.zip"
    
    if not zip_path.exists():
        print("📥 Downloading EUROSAT dataset...")
        downloaded = False
        
        for url in urls:
            try:
                # Disable SSL verification if needed
                response = requests.get(url, stream=True, verify=False, timeout=30)
                response.raise_for_status()
                
                total_size = int(response.headers.get('content-length', 0))
                
                with open(zip_path, 'wb') as f:
                    with tqdm(total=total_size, unit='B', unit_scale=True, desc="Downloading") as pbar:
                        for chunk in response.iter_content(32*1024):
                            if chunk:
                                f.write(chunk)
                                pbar.update(len(chunk))
                
                downloaded = True
                print(f"✓ Downloaded from: {url}")
                break
            except Exception as e:
                print(f"✗ Failed to download from {url}: {str(e)}")
                if zip_path.exists():
                    zip_path.unlink()
                continue
        
        if not downloaded:
            print("\n⚠️  Automated download failed. Please download manually:")
            print("   URL: http://madm.dfki.de/files/sentinel/EuroSAT.zip")
            print(f"   Save to: {zip_path}")
            print("\n   Then run the script again.")
            raise FileNotFoundError("Could not download EUROSAT dataset. Please download manually.")
    
    extract_dir = data_dir / "EuroSAT"
    if not extract_dir.exists():
        print("📦 Extracting dataset...")
        import zipfile
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(data_dir)
            print("✓ Extraction complete")
        except zipfile.BadZipFile:
            print("✗ Downloaded file is corrupted. Removing and retrying...")
            zip_path.unlink()
            return download_eurosat(data_dir)
    
    # Find actual data directory
    possible_paths = [
        data_dir / "2750",
        data_dir / "EuroSAT" / "2750",
        data_dir / "EuroSAT" / "allBands",
        data_dir / "EuroSAT" / "EuroSAT" / "allBands",
        data_dir / "EuroSAT_RGB",
    ]
    
    for path in possible_paths:
        if path.exists():
            print(f"✓ Found data at: {path}")
            return str(path)
    
    print("\n⚠️  Data structure not found. Checking directory contents:")
    for item in data_dir.rglob("*"):
        if item.is_dir() and not item.name.startswith('.'):
            print(f"  └─ {item.relative_to(data_dir)}")
    
    raise FileNotFoundError("EUROSAT data structure not found after extraction.")

# ==================== DATASET CLASS WITH TEMPORAL AUGMENTATION ====================
class EuroSATTemporalDataset(Dataset):
    def __init__(self, data_dir, temporal_length=36, missing_rate=0.2, 
                 selected_classes=None, augment_temporal=True):
        self.data_dir = Path(data_dir)
        self.temporal_length = temporal_length
        self.missing_rate = missing_rate
        self.augment_temporal = augment_temporal
        
        all_classes = [d.name for d in self.data_dir.iterdir() if d.is_dir()]
        patterns = ['*.jpg', '*.jpeg', '*.png']
        self.n_bands = 8
        
        if selected_classes:
            self.classes = selected_classes
        else:
            def count_files(c):
                return sum(len(list((self.data_dir / c).glob(p))) for p in patterns)
            class_counts = {c: count_files(c) for c in all_classes}
            self.classes = [c for c, _ in sorted(class_counts.items(), 
                           key=lambda x: x[1], reverse=True)[:10]]
        
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        self.samples = []
        max_per = CONFIG.get('max_per_class', None)
        
        for cls in self.classes:
            cls_dir = self.data_dir / cls
            count = 0
            for p in patterns:
                for img_path in sorted(cls_dir.glob(p)):
                    self.samples.append((str(img_path), self.class_to_idx[cls]))
                    count += 1
                    if max_per and count >= max_per:
                        break
                if max_per and count >= max_per:
                    break
    
    def create_temporal_sequence(self, image):
        # Reshape to get spatial statistics
        C, H, W = image.shape
        
        # Create temporal sequence by adding noise variations
        temporal_seq = []
        for t in range(self.temporal_length):
            # Add small temporal variation (simulating phenological changes)
            noise = np.random.normal(0, 0.02, image.shape)
            time_factor = (t / self.temporal_length)
            variation = image * (1 + 0.1 * np.sin(2 * np.pi * time_factor))
            frame = np.clip(variation + noise, 0, 1)
            
            # Average across spatial dimensions to get spectral signature
            # This simulates extracting features at each timestep
            spectral_sig = frame.mean(axis=(1, 2))  # (C,)
            temporal_seq.append(spectral_sig)
        
        temporal_seq = np.array(temporal_seq)  # (temporal_length, C)
        
        # Create missing data mask
        mask = np.ones(self.temporal_length, dtype=np.float32)
        n_missing = int(self.temporal_length * self.missing_rate)
        if n_missing > 0:
            missing_indices = np.random.choice(self.temporal_length, n_missing, replace=False)
            mask[missing_indices] = 0
            temporal_seq[missing_indices] = 0  # Zero out missing values
        
        return temporal_seq, mask
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        # Load image as RGB
        image = Image.open(img_path).convert('RGB')
        image = np.array(image, dtype=np.float32) / 255.0
        image = image.transpose(2, 0, 1)  # (C, H, W)
        
        # Create temporal sequence and mask
        temporal_seq, mask = self.create_temporal_sequence(image)
        
        # Convert to tensors
        temporal_seq = torch.tensor(temporal_seq, dtype=torch.float32)
        mask = torch.tensor(mask, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.long)
        
        return temporal_seq, mask, label

# ==================== MCTNET ARCHITECTURE ====================
class ECA(nn.Module):
    """Efficient Channel Attention"""
    def __init__(self, channel, kernel_size=3):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: (B, C, T) or (B, C)
        y = self.avg_pool(x)  # (B, C, 1)
        y = self.conv(y.transpose(-1, -2)).transpose(-1, -2)  # (B, C, 1)
        return x * self.sigmoid(y)

class ALPE(nn.Module):
    """Attention-based Learnable Positional Encoding (as per paper)"""
    def __init__(self, d_model, max_len=256):
        super().__init__()
        self.d_model = d_model
        
        # Standard positional encoding
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                            -(np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 1:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
        
        # Learnable components
        self.conv1d = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1)
        self.eca = ECA(d_model)
    
    def forward(self, x, mask):
        
        B, T, D = x.shape
        
        # Get positional encoding
        pos_enc = self.pe[:T, :].unsqueeze(0).expand(B, -1, -1)  # (B, T, D)
        
        # Apply mask to positional encoding
        pos_enc = pos_enc * mask.unsqueeze(-1)  # (B, T, D)
        
        # Conv1d requires (B, D, T)
        pos_enc = pos_enc.permute(0, 2, 1)  # (B, D, T)
        pos_enc = self.conv1d(pos_enc)
        pos_enc = self.eca(pos_enc)
        pos_enc = pos_enc.permute(0, 2, 1)  # (B, T, D)
        
        return pos_enc

class CNNSubModule(nn.Module):
    """1D CNN sub-module for local temporal feature extraction"""
    def __init__(self, d_model, kernel_size=3):
        super().__init__()
        self.conv1 = nn.Conv1d(d_model, d_model, kernel_size, padding=kernel_size//2)
        self.bn1 = nn.BatchNorm1d(d_model)
        self.conv2 = nn.Conv1d(d_model, d_model, kernel_size, padding=kernel_size//2)
        self.bn2 = nn.BatchNorm1d(d_model)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        
        residual = x
        # Conv1d expects (B, C, L) where L is sequence length
        x = x.permute(0, 2, 1)  # (B, d_model, T)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x = x.permute(0, 2, 1)  # (B, T, d_model)
        x = x.to(residual.dtype)
        return self.relu(x + residual)

class TransformerSubModule(nn.Module):

    def __init__(self, d_model, nhead, use_alpe=False):
        super().__init__()
        self.use_alpe = use_alpe
        if use_alpe:
            self.alpe = ALPE(d_model)
        
        # Multi-head self-attention
        self.attention = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        
        # Feed-forward network
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Linear(d_model * 4, d_model)
        )
        self.norm2 = nn.LayerNorm(d_model)
    
    def forward(self, x, mask=None, return_attention=False):
        # ... positional encoding code ...
        
        # Self-attention with weights
        attn_out, attn_weights = self.attention(
            x, x, x, 
            need_weights=return_attention,
            average_attn_weights=False  # Don't average across heads
        )
        
        if return_attention:
            self.attn_weights = attn_weights  # (B, num_heads, T, T)
        
        x = self.norm1(x + attn_out)
        ff_out = self.ff(x)
        x = self.norm2(x + ff_out)
        
        return x
class CTFusion(nn.Module):
    """CNN-Transformer Fusion module"""
    def __init__(self, d_model, nhead, kernel_size=3, use_alpe=False):
        super().__init__()
        self.cnn = CNNSubModule(d_model, kernel_size)
        self.transformer = TransformerSubModule(d_model, nhead, use_alpe)
    
    def forward(self, x, mask=None):
        
        c = self.cnn(x)
        t = self.transformer(x, mask)
        # Element-wise addition for fusion
        return c + t

class MCTNet(nn.Module):
    """Multi-stage CNN-Transformer Network (as per paper)"""
    def __init__(self, n_stages=3, n_classes=10, d_model=3, nhead=4, kernel_size=3):
        super().__init__()
        
        self.n_stages = n_stages
        self.d_model = d_model
        
        # Build CT Fusion stages
        self.stages = nn.ModuleList()
        for i in range(n_stages):
            use_alpe = (i == 0)  # ALPE only in first stage
            self.stages.append(CTFusion(d_model, nhead, kernel_size, use_alpe))
        
        # Global max pooling (as per paper)
        self.global_pool = nn.AdaptiveMaxPool1d(1)
        
        # MLP classifier
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, n_classes)
        )
    
    def forward(self, x, mask=None):
        
        # Pass through CT Fusion stages
        for i, stage in enumerate(self.stages):
            if i == 0:
                x = stage(x, mask)
            else:
                x = stage(x, mask=None)
        
        # Global max pooling: (B, T, d_model) -> (B, d_model)
        x = x.permute(0, 2, 1)  # (B, d_model, T)
        x = self.global_pool(x).squeeze(-1)  # (B, d_model)
        
        # Classification
        out = self.classifier(x)
        return out

# ==================== TRAINING ====================
class Trainer:
    def __init__(self, model, train_loader, val_loader, device, lr=0.001, use_amp=False):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self.criterion = nn.CrossEntropyLoss()
        self.use_amp = (use_amp and device.type == 'cuda')
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)
        self.history = {
            'train_loss': [], 'val_loss': [], 
            'val_oa': [], 'val_kappa': [], 'val_f1': []
        }
    
    def train_epoch(self):
        self.model.train()
        total_loss = 0
        
        for batch in self.train_loader:
            temporal_seq, mask, labels = batch
            temporal_seq = temporal_seq.to(self.device, non_blocking=True)
            mask = mask.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)
            
            self.optimizer.zero_grad(set_to_none=True)
            
            if self.use_amp:
                with torch.cuda.amp.autocast():
                    outputs = self.model(temporal_seq, mask)
                    loss = self.criterion(outputs, labels)
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(temporal_seq, mask)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(self.train_loader)
    
    def evaluate(self):
        self.model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in self.val_loader:
                temporal_seq, mask, labels = batch
                temporal_seq = temporal_seq.to(self.device, non_blocking=True)
                mask = mask.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)
                
                with torch.cuda.amp.autocast(enabled=self.use_amp):
                    outputs = self.model(temporal_seq, mask)
                preds = torch.argmax(outputs, dim=1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        oa = np.mean(np.array(all_preds) == np.array(all_labels))
        kappa = cohen_kappa_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average='macro')
        
        return oa, kappa, f1
    
    def train(self, epochs):
        best_f1 = 0
        patience = 15
        patience_counter = 0
        
        for epoch in range(epochs):
            train_loss = self.train_epoch()
            oa, kappa, f1 = self.evaluate()
            
            self.history['train_loss'].append(train_loss)
            self.history['val_oa'].append(oa)
            self.history['val_kappa'].append(kappa)
            self.history['val_f1'].append(f1)
            
            print(f"Epoch {epoch+1:03d}: Loss={train_loss:.4f}, OA={oa:.4f}, "
                  f"Kappa={kappa:.4f}, F1={f1:.4f}")
            
            if f1 > best_f1:
                best_f1 = f1
                patience_counter = 0
                torch.save(self.model.state_dict(), 'mctnet_best.pth')
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
        
        return self.history

# ==================== VISUALIZATION ====================
def plot_metrics(history):
    """Plot training metrics (Figure from paper)"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 8))
    axes = axes.flatten()
    
    axes[0].plot(history['train_loss'], label='Train Loss', marker='o', linewidth=2)
    axes[0].set_title('Training Loss', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(history['val_oa'], label='OA', marker='o', linewidth=2)
    axes[1].plot(history['val_kappa'], label='Kappa', marker='s', linewidth=2)
    axes[1].set_title('Validation Metrics', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Score', fontsize=12)
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3)
    
    axes[2].plot(history['val_f1'], label='F1 Score', marker='^', color='green', linewidth=2)
    axes[2].set_title('F1 Score', fontsize=14, fontweight='bold')
    axes[2].set_xlabel('Epoch', fontsize=12)
    axes[2].set_ylabel('F1', fontsize=12)
    axes[2].legend(fontsize=11)
    axes[2].grid(True, alpha=0.3)

    
    axes[3].plot(history['val_oa'], label='Accuracy (OA)', marker='o', linewidth=2)
    axes[3].set_title('Accuracy', fontsize=14, fontweight='bold')
    axes[3].set_xlabel('Epoch', fontsize=12)
    axes[3].set_ylabel('Accuracy', fontsize=12)
    axes[3].legend(fontsize=11)
    axes[3].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('training_metrics.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: training_metrics.png")
    plt.show()

def plot_confusion_matrix(y_true, y_pred, classes, title='Confusion Matrix'):
    """Plot normalized confusion matrix (Figure 7-8 in paper)"""
    cm = confusion_matrix(y_true, y_pred)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm_normalized, annot=True, fmt='.3f', xticklabels=classes, 
                yticklabels=classes, cmap='Blues', cbar_kws={'label': 'Normalized Count'},
                annot_kws={"size": 9})
    plt.title(title, fontsize=14, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: confusion_matrix.png")
    plt.show()

def plot_per_class_metrics(y_true, y_pred, classes):
    """Plot per-class precision, recall, F1 (paper analysis)"""
    from sklearn.metrics import precision_recall_fscore_support
    
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average=None)
    
    x = np.arange(len(classes))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.bar(x - width, precision, width, label='Precision', alpha=0.8)
    ax.bar(x, recall, width, label='Recall', alpha=0.8)
    ax.bar(x + width, f1, width, label='F1 Score', alpha=0.8)
    
    ax.set_xlabel('Class', fontsize=12, fontweight='bold')
    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_title('Per-Class Performance Metrics', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=45, ha='right')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim([0, 1.05])
    
    plt.tight_layout()
    plt.savefig('per_class_metrics.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: per_class_metrics.png")
    plt.show()

def extract_cnn_transformer_features(model, data_loader, device, n_samples=500):
    """Extract features from CNN and Transformer sub-modules (like paper Grad-CAM)"""
    model.eval()
    
    cnn_features_list = []
    trans_features_list = []
    labels_list = []
    
    # Hook to extract intermediate features
    cnn_activations = []
    trans_activations = []
    
    def cnn_hook(module, input, output):
        cnn_activations.append(output.detach())
    
    def trans_hook(module, input, output):
        trans_activations.append(output.detach())
    
    # Register hooks on last stage
    last_stage = model.stages[-1]
    last_stage.cnn.register_forward_hook(cnn_hook)
    last_stage.transformer.register_forward_hook(trans_hook)
    
    with torch.no_grad():
        for i, (temporal_seq, mask, labels) in enumerate(data_loader):
            if i * data_loader.batch_size >= n_samples:
                break
                
            temporal_seq = temporal_seq.to(device)
            mask = mask.to(device)
            
            # Forward pass
            _ = model(temporal_seq, mask)
            
            # Extract features
            if cnn_activations:
                cnn_feat = cnn_activations[-1]
                cnn_feat_pooled = cnn_feat.mean(dim=1)  # Average over time
                cnn_features_list.append(cnn_feat_pooled.cpu().numpy())
                cnn_activations.clear()
            
            if trans_activations:
                trans_feat = trans_activations[-1]
                trans_feat_pooled = trans_feat.mean(dim=1)  # Average over time
                trans_features_list.append(trans_feat_pooled.cpu().numpy())
                trans_activations.clear()
            
            labels_list.extend(labels.numpy())
    
    cnn_features = np.vstack(cnn_features_list) if cnn_features_list else None
    trans_features = np.vstack(trans_features_list) if trans_features_list else None
    labels_array = np.array(labels_list[:len(cnn_features)] if cnn_features is not None else [])
    
    return cnn_features, trans_features, labels_array

def plot_tsne_visualization(features, labels, classes, title='t-SNE Visualization'):
    """t-SNE visualization of extracted features (Figure 13-14 in paper)"""
    if features is None or len(features) == 0:
        print("No features to visualize")
        return
    
    print(f"Running t-SNE on {features.shape[0]} samples...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
    features_2d = tsne.fit_transform(features)
    
    plt.figure(figsize=(12, 10))
    colors = plt.cm.tab10(np.linspace(0, 1, len(classes)))
    
    for i, cls in enumerate(classes):
        mask = labels == i
        plt.scatter(features_2d[mask, 0], features_2d[mask, 1], 
                   label=cls, alpha=0.7, s=50, color=colors[i])
    
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel('t-SNE Component 1', fontsize=12)
    plt.ylabel('t-SNE Component 2', fontsize=12)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    filename = title.lower().replace(' ', '_') + '.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {filename}")
    plt.show()

def plot_temporal_attention_heatmap(model, data_loader, device, class_idx, n_samples=3):
    """Visualize temporal attention weights from Transformer (like paper Figure 17)"""
    model.eval()
    
    class_samples = []
    with torch.no_grad():
        for temporal_seq, mask, labels in data_loader:
            for i, label in enumerate(labels):
                if label == class_idx:
                    class_samples.append((temporal_seq[i:i+1], mask[i:i+1]))
                if len(class_samples) >= n_samples:
                    break
            if len(class_samples) >= n_samples:
                break
    
    if not class_samples:
        print(f"No samples found for class {class_idx}")
        return
    
    print(f"Generating attention heatmaps for {len(class_samples)} samples...")
    
    fig, axes = plt.subplots(n_samples, 1, figsize=(14, 4*n_samples))
    if n_samples == 1:
        axes = [axes]
    
    for idx, (temporal_seq, mask) in enumerate(class_samples):
        temporal_seq = temporal_seq.to(device)
        mask = mask.to(device)
        
        # Get attention weights from first Transformer head
        with torch.no_grad():
            # Extract attention from first stage transformer
            first_stage = model.stages[0].transformer
            
            # Forward through first stage
            x = temporal_seq
            for stage in model.stages[:-1]:
                x = stage(x, mask)
            
            # Get final attention
            x_with_pos = x
            attn_output, attn_weights = first_stage.attention(
                x_with_pos, x_with_pos, x_with_pos,
                need_weights=True, average_attn_weights=False
            )
            if attn_weights.dim() == 4:
                attn_weights = attn_weights.mean(dim=1)[0]
            else:
                attn_weights = attn_weights[0]
        
        im = axes[idx].imshow(attn_weights.cpu().numpy(), cmap='hot', aspect='auto')
        axes[idx].set_title(f'Sample {idx+1} - Temporal Attention', fontsize=12, fontweight='bold')
        axes[idx].set_xlabel('Attention to Timestep', fontsize=10)
        axes[idx].set_ylabel('From Timestep', fontsize=10)
        plt.colorbar(im, ax=axes[idx])
        
        # Mark missing values
        missing_mask = (mask[0].cpu().numpy() == 0)
        missing_indices = np.where(missing_mask)[0]
        for mi in missing_indices:
            axes[idx].axvline(x=mi, color='cyan', linestyle='--', alpha=0.5, linewidth=1)
            axes[idx].axhline(y=mi, color='cyan', linestyle='--', alpha=0.5, linewidth=1)
    
    plt.tight_layout()
    plt.savefig('temporal_attention_heatmap.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: temporal_attention_heatmap.png")
    plt.show()

def plot_missing_data_robustness(test_loader, full_dataset, model, device, classes):
    """Test robustness to different missing rates (Figure 15 in paper)"""
    missing_rates = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    accuracies = []
    
    model.eval()
    
    print("Testing robustness to missing data...")
    for missing_rate in missing_rates:
        correct = 0
        total = 0
        
        with torch.no_grad():
            for temporal_seq, original_mask, labels in test_loader:
                # Create new mask with specified missing rate
                B, T = original_mask.shape
                new_mask = torch.ones_like(original_mask)
                n_missing = int(T * missing_rate)
                
                for b in range(B):
                    if n_missing > 0:
                        missing_indices = np.random.choice(T, n_missing, replace=False)
                        new_mask[b, missing_indices] = 0
                        temporal_seq[b, missing_indices, :] = 0
                
                temporal_seq = temporal_seq.to(device)
                new_mask = new_mask.to(device)
                labels = labels.to(device)
                
                outputs = model(temporal_seq, new_mask)
                preds = torch.argmax(outputs, dim=1)
                
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        
        acc = correct / total
        accuracies.append(acc)
        print(f"  Missing rate {missing_rate:.1%}: OA = {acc:.4f}")
    
    plt.figure(figsize=(10, 6))
    plt.plot(missing_rates, accuracies, marker='o', linewidth=2, markersize=8, color='#2E86AB')
    plt.xlabel('Missing Data Rate', fontsize=12, fontweight='bold')
    plt.ylabel('Overall Accuracy', fontsize=12, fontweight='bold')
    plt.title('Model Robustness to Missing Data', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.xticks([r for r in missing_rates], [f'{r:.0%}' for r in missing_rates])
    plt.tight_layout()
    plt.savefig('missing_data_robustness.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: missing_data_robustness.png")
    plt.show()

def plot_temporal_patterns(dataset, classes, n_samples_per_class=5):
    """Plot temporal patterns for each class (like Figure 2 in paper)"""
    fig, axes = plt.subplots(len(classes), 1, figsize=(14, 3*len(classes)))
    if len(classes) == 1:
        axes = [axes]
    
    for cls_idx, cls_name in enumerate(classes):
        class_samples = [i for i, (_, label) in enumerate(dataset.samples) 
                        if label == cls_idx]
        
        temporal_sequences = []
        for sample_idx in class_samples[:n_samples_per_class]:
            temporal_seq, _, _ = dataset[sample_idx]
            temporal_sequences.append(temporal_seq.numpy())
        
        if temporal_sequences:
            temporal_sequences = np.array(temporal_sequences)
            mean_seq = temporal_sequences.mean(axis=0)
            std_seq = temporal_sequences.std(axis=0)
            
            timesteps = np.arange(mean_seq.shape[0])
            for c in range(mean_seq.shape[1]):
                axes[cls_idx].plot(timesteps, mean_seq[:, c], marker='o', 
                                  label=f'Band {c}', linewidth=2, markersize=4)
                axes[cls_idx].fill_between(timesteps, 
                                          mean_seq[:, c] - std_seq[:, c],
                                          mean_seq[:, c] + std_seq[:, c],
                                          alpha=0.2)
            
            axes[cls_idx].set_title(f'{cls_name} - Temporal Pattern', 
                                   fontsize=12, fontweight='bold')
            axes[cls_idx].set_ylabel('Spectral Value', fontsize=10)
            axes[cls_idx].legend(fontsize=9, loc='upper right')
            axes[cls_idx].grid(True, alpha=0.3)
    
    axes[-1].set_xlabel('Timestep', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig('temporal_patterns.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: temporal_patterns.png")
    plt.show()

# ==================== MAIN EXECUTION ====================
def main():
    print(f"Using device: {CONFIG['device']}")
    print(f"Temporal sequence length: {CONFIG['temporal_seq_length']}")
    print(f"Missing data rate: {CONFIG['missing_rate']}")
    
    # Download data
    data_dir = download_eurosat(CONFIG['data_dir'])
    
    # Inspect structure
    inspect_eurosat_structure(data_dir)
    
    # Create dataset with temporal augmentation
    print("\nCreating temporal dataset...")
    full_dataset = EuroSATTemporalDataset(
        data_dir,
        temporal_length=CONFIG['temporal_seq_length'],
        missing_rate=CONFIG['missing_rate']
    )
    print(f"Classes: {full_dataset.classes}")
    print(f"Total samples: {len(full_dataset)}")
    print(f"Number of bands: {full_dataset.n_bands}")
    
    # Split data
    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])
    
    val_size = int(0.2 * len(train_dataset))
    train_size = len(train_dataset) - val_size
    train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])
    
    # Data loaders
    pin = CONFIG['device'].type == 'cuda'
    train_loader = DataLoader(
        train_dataset, batch_size=CONFIG['batch_size'], 
        shuffle=True, num_workers=CONFIG['num_workers'], pin_memory=pin
    )
    val_loader = DataLoader(
        val_dataset, batch_size=CONFIG['batch_size'], 
        num_workers=CONFIG['num_workers'], pin_memory=pin
    )
    test_loader = DataLoader(
        test_dataset, batch_size=CONFIG['batch_size'], 
        num_workers=CONFIG['num_workers'], pin_memory=pin
    )
    
    print(f"\nTrain samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    
    # Model
    nhead = math.gcd(CONFIG['n_heads'], CONFIG['d_model'])
    model = MCTNet(
        n_stages=CONFIG['n_stages'],
        n_classes=CONFIG['n_classes'],
        d_model=CONFIG['d_model'],
        nhead=nhead,
        kernel_size=CONFIG['kernel_size']
    )
    
    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")
    if CONFIG['device'].type == 'cuda':
        torch.backends.cudnn.benchmark = True
    
    # Train
    trainer = Trainer(
        model, train_loader, val_loader, CONFIG['device'], 
        CONFIG['learning_rate'], use_amp=CONFIG.get('use_amp', False)
    )
    history = trainer.train(CONFIG['epochs'])
    
    # Plot metrics
    plot_metrics(history)
    
    # Evaluate on test set
    model.load_state_dict(torch.load('mctnet_best.pth'))
    model.eval()
    
    test_preds = []
    test_labels = []
    
    with torch.no_grad():
        for batch in test_loader:
            temporal_seq, mask, labels = batch
            temporal_seq = temporal_seq.to(CONFIG['device'], non_blocking=True)
            mask = mask.to(CONFIG['device'], non_blocking=True)
            
            outputs = model(temporal_seq, mask)
            preds = torch.argmax(outputs, dim=1)
            
            test_preds.extend(preds.cpu().numpy())
            test_labels.extend(labels.numpy())
    
    # Confusion matrix
    plot_confusion_matrix(test_labels, test_preds, full_dataset.classes[:CONFIG['n_classes']])
    
    # Metrics
    oa = np.mean(np.array(test_preds) == np.array(test_labels))
    kappa = cohen_kappa_score(test_labels, test_preds)
    f1 = f1_score(test_labels, test_preds, average='macro')
    
    print(f"\n{'='*50}")
    print(f"Test Performance:")
    print(f"Overall Accuracy: {oa:.4f}")
    print(f"Kappa Coefficient: {kappa:.4f}")
    print(f"Macro F1 Score: {f1:.4f}")
    print(f"{'='*50}\n")
    
    # ========== COMPREHENSIVE VISUALIZATIONS (FROM PAPER) ==========
    print("Generating visualizations...")
    
    # 1. Per-class metrics
    print("\n1. Per-Class Performance Metrics")
    plot_per_class_metrics(test_labels, test_preds, full_dataset.classes)
    
    # 2. t-SNE visualization of CNN features
    print("\n2. Extracting and visualizing features...")
    cnn_features, trans_features, feature_labels = extract_cnn_transformer_features(
        model, test_loader, CONFIG['device'], n_samples=500
    )
    
    if cnn_features is not None:
        print("3. CNN Sub-module Features (t-SNE)")
        plot_tsne_visualization(cnn_features, feature_labels, full_dataset.classes,
                               title='CNN Sub-module Features (t-SNE)')
    
    if trans_features is not None:
        print("4. Transformer Sub-module Features (t-SNE)")
        plot_tsne_visualization(trans_features, feature_labels, full_dataset.classes,
                               title='Transformer Sub-module Features (t-SNE)')
    
    # 3. Temporal attention heatmaps
    print("\n5. Temporal Attention Visualization")
    for cls_idx in range(min(3, CONFIG['n_classes'])):
        print(f"   Generating attention for class {full_dataset.classes[cls_idx]}...")
        plot_temporal_attention_heatmap(model, test_loader, CONFIG['device'], 
                                       class_idx=cls_idx, n_samples=2)
    
    # 4. Robustness to missing data
    print("\n6. Testing Robustness to Missing Data")
    plot_missing_data_robustness(test_loader, full_dataset, model, 
                                CONFIG['device'], full_dataset.classes)
    
    # 5. Temporal patterns per class
    print("\n7. Temporal Patterns by Class")
    plot_temporal_patterns(full_dataset, full_dataset.classes, n_samples_per_class=5)
    
    print("✓ All visualizations completed!")
    print("\nGenerated files:")
    print("  • training_metrics.png - Training history")
    print("  • confusion_matrix.png - Normalized confusion matrix")
    print("  • per_class_metrics.png - Per-class precision/recall/F1")
    print("  • cnn_sub_module_features_(t-sne_visualization).png - CNN features")
    print("  • transformer_sub_module_features_(t-sne_visualization).png - Transformer features")
    print("  • temporal_attention_heatmap.png - Attention weights over time")
    print("  • missing_data_robustness.png - Robustness curve")
    print("  • temporal_patterns.png - Class-specific temporal dynamics")
    print("="*60)

if __name__ == '__main__':
    main()