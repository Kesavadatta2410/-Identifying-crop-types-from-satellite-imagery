import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
from pathlib import Path
import numpy as np
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_fscore_support, cohen_kappa_score
from sklearn.manifold import TSNE

# ==================== CONFIGURATION ====================
CONFIG = {
    'data_dir': './eurosat_data',
    'batch_size': 32,
    'epochs': 5,
    'learning_rate': 0.001,
    'num_workers': 0, # Set to 0 to avoid Windows multiprocessing issues
    'device': torch.device('cuda'),
    'image_size': 224, # EfficientNet-B0 input size
    'results_dir': './results_transfer',
    'n_classes': 10,
    'max_per_class': 2500,
    'use_amp': True,  # Automatic Mixed Precision for faster training
    'dropout_rate': 0.3,
    'weight_decay': 1e-4,
    # Temporal augmentation settings (from paper.py)
    'use_temporal_augmentation': True,
    'temporal_seq_length': 8,  # Number of temporal augmentations per image
    'temporal_noise_std': 0.02,  # Noise standard deviation for temporal variations
    'temporal_variation_factor': 0.1,  # Variation factor for simulating phenological changes
    # Additional noise augmentation
    'gaussian_noise_std': 0.01,  # Gaussian noise added to images
    'salt_pepper_prob': 0.01,  # Salt and pepper noise probability
}

# Ensure results directory exists
os.makedirs(CONFIG['results_dir'], exist_ok=True)

# ==================== TEMPORAL AUGMENTATION UTILITIES ====================
def add_gaussian_noise(image, std=0.01):
    """Add Gaussian noise to image (from paper.py style)"""
    noise = np.random.normal(0, std, image.shape)
    return np.clip(image + noise, 0, 1)

def add_salt_pepper_noise(image, prob=0.01):
    """Add salt and pepper noise to image"""
    noisy = image.copy()
    # Salt noise
    salt_mask = np.random.random(image.shape) < (prob / 2)
    noisy[salt_mask] = 1
    # Pepper noise
    pepper_mask = np.random.random(image.shape) < (prob / 2)
    noisy[pepper_mask] = 0
    return noisy

def create_temporal_augmentations(image, seq_length=8, noise_std=0.02, variation_factor=0.1):
    """
    Create temporal sequence augmentations from a single image (inspired by paper.py).
    Simulates temporal variations like phenological changes.
    
    Args:
        image: numpy array of shape (C, H, W)
        seq_length: number of temporal frames to generate
        noise_std: standard deviation of temporal noise
        variation_factor: factor for simulating temporal variations
    
    Returns:
        List of augmented images
    """
    temporal_sequence = []
    for t in range(seq_length):
        # Add temporal variation (simulating phenological changes)
        time_factor = (t / seq_length)
        variation = image * (1 + variation_factor * np.sin(2 * np.pi * time_factor))
        
        # Add temporal noise
        noise = np.random.normal(0, noise_std, image.shape)
        augmented = np.clip(variation + noise, 0, 1)
        
        temporal_sequence.append(augmented)
    
    return temporal_sequence

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
            # Fallback
            for item in self.data_dir.rglob("*"):
                if item.is_dir() and not item.name.startswith('.'):
                    if list(item.glob("*.jpg")) or list(item.glob("*.png")):
                        self.actual_data_dir = item.parent
                        break
        
        if self.actual_data_dir is None:
             raise FileNotFoundError(f"Could not find EuroSAT data in {data_dir}")

        self.classes = sorted([d.name for d in self.actual_data_dir.iterdir() if d.is_dir()])
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        self.samples = []
        
        patterns = ['*.jpg', '*.jpeg', '*.png']
        for cls in self.classes:
            cls_dir = self.actual_data_dir / cls
            for p in patterns:
                for img_path in sorted(cls_dir.glob(p)):
                    self.samples.append((str(img_path), self.class_to_idx[cls]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        
        # Apply temporal augmentation if enabled (before transform)
        if CONFIG.get('use_temporal_augmentation', False) and not self.transform:
            # Convert PIL to numpy for temporal augmentation
            img_array = np.array(image, dtype=np.float32) / 255.0
            img_array = img_array.transpose(2, 0, 1)  # (C, H, W)
            
            # Create temporal sequence
            temporal_seq = create_temporal_augmentations(
                img_array,
                seq_length=CONFIG.get('temporal_seq_length', 8),
                noise_std=CONFIG.get('temporal_noise_std', 0.02),
                variation_factor=CONFIG.get('temporal_variation_factor', 0.1)
            )
            
            # Return random sample from temporal sequence
            selected_frame = temporal_seq[np.random.randint(len(temporal_seq))]
            
            # Add additional noise augmentations
            if CONFIG.get('gaussian_noise_std', 0) > 0:
                selected_frame = add_gaussian_noise(selected_frame, CONFIG['gaussian_noise_std'])
            if CONFIG.get('salt_pepper_prob', 0) > 0:
                selected_frame = add_salt_pepper_noise(selected_frame, CONFIG['salt_pepper_prob'])
            
            # Convert back to PIL Image
            selected_frame = (selected_frame.transpose(1, 2, 0) * 255).astype(np.uint8)
            image = Image.fromarray(selected_frame)
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

# ==================== MODEL ====================
def get_transfer_model(num_classes, dropout_rate=0.3):
    # Use EfficientNet-B0
    print("Loading EfficientNet-B0...")
    model = models.efficientnet_b0(pretrained=True)
    
    # Replace the classifier head with dropout for regularization
    # EfficientNet's classifier is a Sequential block, usually ending with a Linear layer
    # We need to find the input features of the last linear layer
    num_ftrs = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=dropout_rate, inplace=True),  # Added dropout to prevent overfitting
        nn.Linear(num_ftrs, num_classes)
    )
    
    return model

# ==================== VISUALIZATIONS ====================
def plot_metrics(history):
    """Plot training metrics (enhanced style from paper)"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 8))
    axes = axes.flatten()
    
    # Training Loss
    axes[0].plot(history['train_loss'], label='Train Loss', marker='o', linewidth=2)
    axes[0].plot(history['val_loss'], label='Val Loss', marker='s', linewidth=2)
    axes[0].set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)
    
    # Accuracy
    axes[1].plot(history['train_acc'], label='Train Acc', marker='o', linewidth=2)
    axes[1].plot(history['val_acc'], label='Val Acc', marker='s', linewidth=2)
    axes[1].set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Accuracy', fontsize=12)
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3)
    
    # F1 Score (if available)
    if 'val_f1' in history and history['val_f1']:
        axes[2].plot(history['val_f1'], label='F1 Score', marker='^', color='green', linewidth=2)
        axes[2].set_title('Validation F1 Score', fontsize=14, fontweight='bold')
        axes[2].set_xlabel('Epoch', fontsize=12)
        axes[2].set_ylabel('F1 Score', fontsize=12)
        axes[2].legend(fontsize=11)
        axes[2].grid(True, alpha=0.3)
    else:
        axes[2].axis('off')
    
    # Learning Rate
    if 'learning_rate' in history and history['learning_rate']:
        axes[3].plot(history['learning_rate'], label='Learning Rate', marker='d', color='orange', linewidth=2)
        axes[3].set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
        axes[3].set_xlabel('Epoch', fontsize=12)
        axes[3].set_ylabel('Learning Rate', fontsize=12)
        axes[3].set_yscale('log')
        axes[3].legend(fontsize=11)
        axes[3].grid(True, alpha=0.3)
    else:
        axes[3].axis('off')
    
    plt.tight_layout()
    save_path = os.path.join(CONFIG['results_dir'], 'training_metrics.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {save_path}")
    plt.close()

def plot_confusion_matrix(y_true, y_pred, classes, title='Confusion Matrix (Transfer Learning)'):
    """Plot normalized confusion matrix (enhanced style from paper)"""
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
    
    save_path = os.path.join(CONFIG['results_dir'], 'confusion_matrix.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {save_path}")
    plt.close()

def plot_per_class_metrics(y_true, y_pred, classes):
    """Plot per-class precision, recall, F1 (enhanced style from paper)"""
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average=None)
    
    x = np.arange(len(classes))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.bar(x - width, precision, width, label='Precision', alpha=0.8)
    ax.bar(x, recall, width, label='Recall', alpha=0.8)
    ax.bar(x + width, f1, width, label='F1 Score', alpha=0.8)
    
    ax.set_xlabel('Class', fontsize=12, fontweight='bold')
    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_title('Per-Class Performance Metrics (Transfer Learning)', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=45, ha='right')
    ax.legend(fontsize=11)
    ax.grid(True, axis='y', alpha=0.3)
    ax.set_ylim([0, 1.05])
    
    plt.tight_layout()
    save_path = os.path.join(CONFIG['results_dir'], 'per_class_metrics.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {save_path}")
    plt.close()

def plot_tsne(model, loader, device, classes, max_samples=1000):
    """Extract features and plot t-SNE (enhanced style from paper)"""
    print("Extracting features for t-SNE...")
    model.eval()
    features = []
    labels = []
    
    # Hook to get features before classifier
    # EfficientNet features are usually from the avgpool layer
    # We can just use the model.features output and pool it
    
    with torch.no_grad():
        for i, (images, target) in enumerate(loader):
            if i * loader.batch_size >= max_samples:
                break
            images = images.to(device)
            
            # EfficientNet forward pass parts
            x = model.features(images)
            x = model.avgpool(x)
            x = torch.flatten(x, 1)
            
            features.append(x.cpu().numpy())
            labels.append(target.numpy())
            
    features = np.vstack(features)
    labels = np.concatenate(labels)
    
    print(f"Running t-SNE on {len(features)} samples...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
    features_2d = tsne.fit_transform(features)
    
    plt.figure(figsize=(12, 10))
    colors = plt.cm.tab10(np.linspace(0, 1, len(classes)))
    
    for i, cls in enumerate(classes):
        mask = labels == i
        plt.scatter(features_2d[mask, 0], features_2d[mask, 1], 
                   label=cls, alpha=0.7, s=50, color=colors[i])
    
    plt.title('t-SNE Visualization of EfficientNet Features (Transfer Learning)', fontsize=14, fontweight='bold')
    plt.xlabel('t-SNE Component 1', fontsize=12)
    plt.ylabel('t-SNE Component 2', fontsize=12)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    save_path = os.path.join(CONFIG['results_dir'], 'tsne_visualization.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {save_path}")
    plt.close()

def plot_temporal_patterns(dataset, classes, n_samples_per_class=5):
    """Plot temporal patterns for each class (adapted from paper.py)"""
    if not CONFIG.get('use_temporal_augmentation', False):
        print("⚠ Temporal augmentation disabled - skipping temporal patterns plot")
        return
    
    print("Generating temporal patterns visualization...")
    fig, axes = plt.subplots(len(classes), 1, figsize=(14, 3*len(classes)))
    if len(classes) == 1:
        axes = [axes]
    
    for cls_idx, cls_name in enumerate(classes):
        # Get samples for this class
        class_samples = [i for i, (_, label) in enumerate(dataset.samples) 
                        if label == cls_idx]
        
        if not class_samples:
            axes[cls_idx].text(0.5, 0.5, f'No samples for {cls_name}', 
                             ha='center', va='center', fontsize=12)
            axes[cls_idx].set_title(f'{cls_name} - Temporal Pattern', 
                                   fontsize=12, fontweight='bold')
            continue
        
        temporal_sequences = []
        for sample_idx in class_samples[:n_samples_per_class]:
            img_path, _ = dataset.samples[sample_idx]
            try:
                # Load and process image
                image = Image.open(img_path).convert('RGB')
                img_array = np.array(image, dtype=np.float32) / 255.0
                img_array = img_array.transpose(2, 0, 1)  # (C, H, W)
                
                # Create temporal sequence
                temporal_seq = create_temporal_augmentations(
                    img_array,
                    seq_length=CONFIG.get('temporal_seq_length', 8),
                    noise_std=CONFIG.get('temporal_noise_std', 0.02),
                    variation_factor=CONFIG.get('temporal_variation_factor', 0.1)
                )
                
                # Extract mean values across spatial dimensions for each channel
                seq_means = np.array([frame.mean(axis=(1, 2)) for frame in temporal_seq])
                temporal_sequences.append(seq_means)
            except Exception as e:
                print(f"Error processing sample {sample_idx}: {e}")
                continue
        
        if temporal_sequences:
            temporal_sequences = np.array(temporal_sequences)  # (n_samples, seq_length, channels)
            mean_seq = temporal_sequences.mean(axis=0)  # (seq_length, channels)
            std_seq = temporal_sequences.std(axis=0)
            
            timesteps = np.arange(mean_seq.shape[0])
            channel_names = ['Red', 'Green', 'Blue']
            colors_plot = ['red', 'green', 'blue']
            
            for c in range(min(3, mean_seq.shape[1])):
                axes[cls_idx].plot(timesteps, mean_seq[:, c], marker='o', 
                                  label=channel_names[c], linewidth=2, markersize=4,
                                  color=colors_plot[c], alpha=0.8)
                axes[cls_idx].fill_between(timesteps, 
                                          mean_seq[:, c] - std_seq[:, c],
                                          mean_seq[:, c] + std_seq[:, c],
                                          alpha=0.2, color=colors_plot[c])
            
            axes[cls_idx].set_title(f'{cls_name} - Temporal Pattern', 
                                   fontsize=12, fontweight='bold')
            axes[cls_idx].set_ylabel('Channel Intensity', fontsize=10)
            axes[cls_idx].legend(fontsize=9, loc='upper right')
            axes[cls_idx].grid(True, alpha=0.3)
            axes[cls_idx].set_ylim([0, 1])
    
    axes[-1].set_xlabel('Temporal Step', fontsize=12, fontweight='bold')
    plt.tight_layout()
    save_path = os.path.join(CONFIG['results_dir'], 'temporal_patterns.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {save_path}")
    plt.close()

def plot_feature_attention_heatmap(model, loader, device, classes, n_samples=3):
    """Plot feature attention heatmaps using Grad-CAM style visualization (adapted for CNNs)"""
    print("Generating feature attention heatmaps...")
    model.eval()
    
    # Collect samples from different classes
    samples_collected = []
    target_classes = list(range(min(3, len(classes))))  # First 3 classes
    
    with torch.no_grad():
        for images, labels in loader:
            for i, label in enumerate(labels):
                if label.item() in target_classes and len([s for s in samples_collected if s[2] == label.item()]) < n_samples:
                    samples_collected.append((images[i:i+1], label.item(), classes[label.item()]))
                if len(samples_collected) >= n_samples * len(target_classes):
                    break
            if len(samples_collected) >= n_samples * len(target_classes):
                break
    
    if not samples_collected:
        print("⚠ Not enough samples for feature attention visualization")
        return
    
    n_show = min(len(samples_collected), 6)
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for idx in range(n_show):
        if idx >= len(samples_collected):
            axes[idx].axis('off')
            continue
            
        image_tensor, label, class_name = samples_collected[idx]
        image_tensor = image_tensor.to(device)
        
        # Get feature maps from last convolutional layer
        with torch.no_grad():
            features = model.features(image_tensor)
        
        # Compute attention as mean across channels
        attention_map = features.mean(dim=1).squeeze().cpu().numpy()
        
        # Normalize attention map
        attention_map = (attention_map - attention_map.min()) / (attention_map.max() - attention_map.min() + 1e-8)
        
        # Display
        im = axes[idx].imshow(attention_map, cmap='hot', interpolation='bilinear')
        axes[idx].set_title(f'{class_name}', fontsize=11, fontweight='bold')
        axes[idx].axis('off')
        plt.colorbar(im, ax=axes[idx], fraction=0.046, pad=0.04)
    
    # Hide extra subplots
    for idx in range(n_show, len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle('Feature Attention Maps (CNN Features)', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    save_path = os.path.join(CONFIG['results_dir'], 'feature_attention_heatmap.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {save_path}")
    plt.close()

# ==================== TRAINING ====================
def train_model():
    print(f"Using device: {CONFIG['device']}")
    
    # Transforms
    train_transform = transforms.Compose([
        transforms.Resize((CONFIG['image_size'], CONFIG['image_size'])),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((CONFIG['image_size'], CONFIG['image_size'])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Dataset
    try:
        full_dataset = EuroSATDataset(CONFIG['data_dir'], transform=None)
    except FileNotFoundError as e:
        print(e)
        return

    # Split
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    # Helper wrapper to apply transforms
    class TransformDataset(Dataset):
        def __init__(self, subset, transform):
            self.subset = subset
            self.transform = transform
        def __getitem__(self, index):
            x, y = self.subset[index]
            if self.transform:
                x = self.transform(x)
            return x, y
        def __len__(self):
            return len(self.subset)
            
    train_dataset = TransformDataset(train_dataset, train_transform)
    val_dataset = TransformDataset(val_dataset, val_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True, num_workers=CONFIG['num_workers'])
    val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'], shuffle=False, num_workers=CONFIG['num_workers'])
    
    # Model
    model = get_transfer_model(len(full_dataset.classes), dropout_rate=CONFIG['dropout_rate'])
    model = model.to(CONFIG['device'])
    
    criterion = nn.CrossEntropyLoss()
    # Added weight_decay (L2 regularization) to prevent overfitting
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG['learning_rate'], weight_decay=CONFIG['weight_decay'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=3, verbose=True)
    
    # AMP setup (from paper.py)
    use_amp = CONFIG.get('use_amp', False) and CONFIG['device'].type == 'cuda'
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [], 
               'val_f1': [], 'val_kappa': [], 'learning_rate': []}
    best_acc = 0.0
    patience_counter = 0
    early_stopping_patience = 7  # Stop if no improvement for 7 epochs
    
    for epoch in range(CONFIG['epochs']):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{CONFIG['epochs']}")
        for images, labels in pbar:
            images, labels = images.to(CONFIG['device']), labels.to(CONFIG['device'])
            
            optimizer.zero_grad(set_to_none=True)
            
            # Use AMP if enabled (from paper.py)
            if use_amp:
                with torch.cuda.amp.autocast():
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            pbar.set_postfix({'loss': running_loss/total, 'acc': correct/total})            
        train_loss = running_loss / len(train_loader)
        train_acc = correct / total
        
        # Validation
        model.eval()
        val_running_loss = 0.0
        val_correct = 0
        val_total = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(CONFIG['device']), labels.to(CONFIG['device'])
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        val_loss = val_running_loss / len(val_loader)
        val_acc = val_correct / val_total
        
        # Calculate F1 and Kappa (from paper.py)
        val_f1 = f1_score(all_labels, all_preds, average='macro')
        val_kappa = cohen_kappa_score(all_labels, all_preds)
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_f1'].append(val_f1)
        history['val_kappa'].append(val_kappa)
        history['learning_rate'].append(optimizer.param_groups[0]['lr'])
        
        print(f"Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, F1: {val_f1:.4f}, Kappa: {val_kappa:.4f}")
        
        scheduler.step(val_acc)
        
        # Early stopping logic
        if val_acc > best_acc:
            best_acc = val_acc
            patience_counter = 0  # Reset patience counter
            torch.save(model.state_dict(), os.path.join(CONFIG['results_dir'], 'best_transfer_model.pth'))
            print(f"Saved best model with Val Acc: {best_acc:.4f}")
            
            # Save best predictions for final plots
            best_preds = all_preds
            best_labels = all_labels
        else:
            patience_counter += 1
            print(f"No improvement. Patience: {patience_counter}/{early_stopping_patience}")
            
        # Early stopping check
        if patience_counter >= early_stopping_patience:
            print(f"\nEarly stopping triggered after {epoch+1} epochs!")
            break

    # Final Visualizations
    print("\nGenerating visualizations...")
    plot_metrics(history)
    plot_confusion_matrix(best_labels, best_preds, full_dataset.classes)
    plot_per_class_metrics(best_labels, best_preds, full_dataset.classes)
    
    # Load best model for additional visualizations
    model.load_state_dict(torch.load(os.path.join(CONFIG['results_dir'], 'best_transfer_model.pth')))
    plot_tsne(model, val_loader, CONFIG['device'], full_dataset.classes)
    
    # New visualizations (from paper.py style)
    plot_temporal_patterns(full_dataset, full_dataset.classes, n_samples_per_class=5)
    plot_feature_attention_heatmap(model, val_loader, CONFIG['device'], full_dataset.classes, n_samples=2)

if __name__ == "__main__":
    train_model()
