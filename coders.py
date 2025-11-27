import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split, Dataset
from torchvision import transforms
from PIL import Image
from pathlib import Path
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import os

# ==================== CONFIGURATION ====================
CONFIG = {
    'data_dir': './eurosat_data',
    'batch_size': 64,
    'epochs': 20,
    'learning_rate': 0.0005,
    'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    'image_size': 64, 
    'patch_size': 8,
    'embed_dim': 128,
    'num_heads': 4,
    'num_layers': 4,
    'num_workers': 0,
    'results_dir': './results_coders',
}

os.makedirs(CONFIG['results_dir'], exist_ok=True)

# ==================== DATASET ====================
class EuroSATDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = Path(data_dir)
        self.transform = transform
        
        # Find data directory
        possible_paths = [
            self.data_dir / "2750",
            self.data_dir / "EuroSAT" / "2750",
            self.data_dir / "EuroSAT" / "allBands",
            self.data_dir / "EuroSAT" / "EuroSAT" / "allBands",
            self.data_dir / "EuroSAT_RGB",
        ]
        
        self.actual_data_dir = None
        for path in possible_paths:
            if path.exists():
                self.actual_data_dir = path
                break
        
        if self.actual_data_dir is None:
            for item in self.data_dir.rglob("*"):
                if item.is_dir() and not item.name.startswith('.'):
                    if list(item.glob("*.jpg")) or list(item.glob("*.png")):
                        self.actual_data_dir = item.parent
                        break
        
        if self.actual_data_dir is None:
             raise FileNotFoundError(f"Could not find EuroSAT data in {data_dir}")

        self.classes = sorted([d.name for d in self.actual_data_dir.iterdir() if d.is_dir()])
        self.samples = []
        
        patterns = ['*.jpg', '*.jpeg', '*.png']
        for cls in self.classes:
            cls_dir = self.actual_data_dir / cls
            for p in patterns:
                for img_path in sorted(cls_dir.glob(p)):
                    self.samples.append(str(img_path))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        return image

# ==================== TRANSFORMER AUTOENCODER ====================
class PatchEmbedding(nn.Module):
    def __init__(self, image_size, patch_size, embed_dim):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        self.projection = nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # x: (B, 3, H, W)
        x = self.projection(x)  # (B, E, H/P, W/P)
        x = x.flatten(2)        # (B, E, N)
        x = x.transpose(1, 2)   # (B, N, E)
        return x

class TransformerAutoencoder(nn.Module):
    def __init__(self, image_size=64, patch_size=8, embed_dim=128, num_heads=4, num_layers=4):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.num_patches = (image_size // patch_size) ** 2
        
        # Encoder
        self.patch_embed = PatchEmbedding(image_size, patch_size, embed_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Decoder
        # We use a similar transformer structure for decoder
        decoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, batch_first=True)
        self.decoder = nn.TransformerEncoder(decoder_layer, num_layers=num_layers)
        
        # Output projection to pixels
        self.output_head = nn.Linear(embed_dim, patch_size * patch_size * 3)

    def forward(self, x):
        B, _, H, W = x.shape
        
        # Encoder
        x_emb = self.patch_embed(x)
        x_emb = x_emb + self.pos_embed
        latent = self.encoder(x_emb)
        
        # Decoder
        rec_tokens = self.decoder(latent)
        
        # Reconstruct patches
        patches = self.output_head(rec_tokens) # (B, N, P*P*3)
        
        # Reshape patches back to image
        # (B, H/P * W/P, P*P*3) -> (B, 3, H, W)
        rec_img = self.patches_to_image(patches)
        
        return rec_img

    def patches_to_image(self, patches):
        B = patches.shape[0]
        h = w = self.image_size // self.patch_size
        c = 3
        p = self.patch_size
        
        patches = patches.reshape(B, h, w, p, p, c)
        patches = patches.permute(0, 5, 1, 3, 2, 4)
        img = patches.reshape(B, c, h * p, w * p)
        return img

# ==================== VISUALIZATIONS ====================
def plot_reconstructions(model, loader, device, n_samples=5):
    """Plot original vs reconstructed images"""
    model.eval()
    images = next(iter(loader))
    images = images[:n_samples].to(device)
    
    with torch.no_grad():
        recons = model(images)
    
    images = images.cpu().permute(0, 2, 3, 1).numpy()
    recons = recons.cpu().permute(0, 2, 3, 1).numpy()
    
    # Clip to 0-1
    recons = np.clip(recons, 0, 1)
    
    fig, axes = plt.subplots(2, n_samples, figsize=(n_samples * 3, 6))
    for i in range(n_samples):
        axes[0, i].imshow(images[i])
        axes[0, i].set_title("Original")
        axes[0, i].axis('off')
        
        axes[1, i].imshow(recons[i])
        axes[1, i].set_title("Reconstructed")
        axes[1, i].axis('off')
        
    plt.tight_layout()
    save_path = os.path.join(CONFIG['results_dir'], 'reconstructions.png')
    plt.savefig(save_path)
    print(f"Saved reconstruction plot to {save_path}")
    plt.close()

def plot_loss(history):
    """Plot training loss"""
    plt.figure(figsize=(10, 6))
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.title('Autoencoder Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.legend()
    plt.grid(True)
    
    save_path = os.path.join(CONFIG['results_dir'], 'training_loss.png')
    plt.savefig(save_path)
    print(f"Saved loss plot to {save_path}")
    plt.close()

# ==================== TRAINING ====================
def train_autoencoder():
    print(f"Using device: {CONFIG['device']}")
    
    transform = transforms.Compose([
        transforms.Resize((CONFIG['image_size'], CONFIG['image_size'])),
        transforms.ToTensor(),
    ])
    
    try:
        dataset = EuroSATDataset(CONFIG['data_dir'], transform=transform)
    except FileNotFoundError:
        print("Dataset not found.")
        return

    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True, num_workers=CONFIG['num_workers'])
    val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'], shuffle=False, num_workers=CONFIG['num_workers'])
    
    model = TransformerAutoencoder(
        image_size=CONFIG['image_size'],
        patch_size=CONFIG['patch_size'],
        embed_dim=CONFIG['embed_dim'],
        num_heads=CONFIG['num_heads'],
        num_layers=CONFIG['num_layers']
    ).to(CONFIG['device'])
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG['learning_rate'])
    
    history = {'train_loss': [], 'val_loss': []}
    best_loss = float('inf')
    
    for epoch in range(CONFIG['epochs']):
        model.train()
        running_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{CONFIG['epochs']}")
        for images in pbar:
            images = images.to(CONFIG['device'])
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, images)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            pbar.set_postfix({'loss': running_loss/len(train_loader)})
            
        train_loss = running_loss / len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images in val_loader:
                images = images.to(CONFIG['device'])
                outputs = model(images)
                loss = criterion(outputs, images)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        
        print(f"Epoch {epoch+1}: Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
        
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), os.path.join(CONFIG['results_dir'], 'best_autoencoder.pth'))
            print("Saved best model")

    # Visualizations
    print("\nGenerating visualizations...")
    plot_loss(history)
    
    # Load best model for reconstructions
    model.load_state_dict(torch.load(os.path.join(CONFIG['results_dir'], 'best_autoencoder.pth')))
    plot_reconstructions(model, val_loader, CONFIG['device'])

if __name__ == "__main__":
    train_autoencoder()
