---
title: "Computer Vision with Deep Learning: From CNNs to Vision Transformers"
date: "2024-01-12"
author: "Dr. James Liu"
excerpt: "Explore the evolution of computer vision from traditional methods to modern deep learning approaches."
tags: ["computer-vision", "deep-learning", "cnn", "transformers", "pytorch"]
category: "Computer Vision"
---

# Computer Vision with Deep Learning: From CNNs to Vision Transformers

Computer Vision has undergone a revolutionary transformation with the advent of deep learning. This comprehensive guide explores the journey from traditional computer vision techniques to state-of-the-art deep learning models.

## Introduction to Computer Vision

Computer Vision enables machines to interpret and understand visual information from the world. Modern deep learning approaches have achieved human-level performance in many vision tasks.

### Key Applications:
- **Image Classification**: Categorizing images into predefined classes
- **Object Detection**: Locating and identifying objects in images
- **Semantic Segmentation**: Pixel-level classification
- **Face Recognition**: Identifying individuals from facial features
- **Medical Imaging**: Analyzing X-rays, MRIs, and CT scans

## Convolutional Neural Networks (CNNs)

CNNs are the foundation of modern computer vision, designed to process grid-like data such as images.

### CNN Architecture Components

#### 1. Convolutional Layers
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

# Example usage
conv_block = ConvBlock(3, 64)  # RGB input to 64 feature maps
```

#### 2. Pooling Layers
```python
# Max pooling reduces spatial dimensions
max_pool = nn.MaxPool2d(kernel_size=2, stride=2)

# Average pooling
avg_pool = nn.AvgPool2d(kernel_size=2, stride=2)

# Adaptive pooling (output size independent of input size)
adaptive_pool = nn.AdaptiveAvgPool2d((7, 7))
```

### Classic CNN Architectures

#### LeNet-5 (1998)
```python
class LeNet5(nn.Module):
    def __init__(self, num_classes=10):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)
    
    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
```

#### ResNet (2015)
```python
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(residual)
        out = F.relu(out)
        return out
```

## Object Detection

### YOLO (You Only Look Once)
```python
class YOLOv5(nn.Module):
    def __init__(self, num_classes=80):
        super(YOLOv5, self).__init__()
        self.num_classes = num_classes
        # Simplified YOLOv5 structure
        self.backbone = self._make_backbone()
        self.neck = self._make_neck()
        self.head = self._make_head()
    
    def _make_backbone(self):
        # CSPDarknet backbone
        layers = []
        # Implementation details...
        return nn.Sequential(*layers)
    
    def forward(self, x):
        # Extract features
        features = self.backbone(x)
        # Feature pyramid network
        fpn_features = self.neck(features)
        # Detection head
        predictions = self.head(fpn_features)
        return predictions

# Non-Maximum Suppression
def nms(boxes, scores, iou_threshold=0.5):
    """
    Apply Non-Maximum Suppression to remove overlapping boxes
    """
    indices = torch.argsort(scores, descending=True)
    keep = []
    
    while len(indices) > 0:
        current = indices[0]
        keep.append(current)
        
        if len(indices) == 1:
            break
            
        current_box = boxes[current]
        remaining_boxes = boxes[indices[1:]]
        
        # Calculate IoU
        ious = calculate_iou(current_box, remaining_boxes)
        
        # Keep boxes with IoU less than threshold
        indices = indices[1:][ious < iou_threshold]
    
    return torch.tensor(keep)
```

## Semantic Segmentation

### U-Net Architecture
```python
class UNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=1):
        super(UNet, self).__init__()
        
        # Encoder (Contracting Path)
        self.enc1 = self._make_encoder_block(in_channels, 64)
        self.enc2 = self._make_encoder_block(64, 128)
        self.enc3 = self._make_encoder_block(128, 256)
        self.enc4 = self._make_encoder_block(256, 512)
        
        # Bottleneck
        self.bottleneck = self._make_encoder_block(512, 1024)
        
        # Decoder (Expanding Path)
        self.dec4 = self._make_decoder_block(1024, 512)
        self.dec3 = self._make_decoder_block(512, 256)
        self.dec2 = self._make_decoder_block(256, 128)
        self.dec1 = self._make_decoder_block(128, 64)
        
        # Final layer
        self.final = nn.Conv2d(64, num_classes, 1)
    
    def _make_encoder_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def _make_decoder_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, 2, stride=2),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(F.max_pool2d(e1, 2))
        e3 = self.enc3(F.max_pool2d(e2, 2))
        e4 = self.enc4(F.max_pool2d(e3, 2))
        
        # Bottleneck
        b = self.bottleneck(F.max_pool2d(e4, 2))
        
        # Decoder with skip connections
        d4 = self.dec4(b)
        d4 = torch.cat([d4, e4], dim=1)
        
        d3 = self.dec3(d4)
        d3 = torch.cat([d3, e3], dim=1)
        
        d2 = self.dec2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        
        d1 = self.dec1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        
        return self.final(d1)
```

## Vision Transformers (ViTs)

### Transformer Architecture for Vision
```python
class PatchEmbedding(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        
        self.projection = nn.Conv2d(in_channels, embed_dim, patch_size, patch_size)
    
    def forward(self, x):
        # x: (batch_size, channels, height, width)
        x = self.projection(x)  # (batch_size, embed_dim, n_patches**0.5, n_patches**0.5)
        x = x.flatten(2)        # (batch_size, embed_dim, n_patches)
        x = x.transpose(1, 2)   # (batch_size, n_patches, embed_dim)
        return x

class VisionTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_channels=3, num_classes=1000,
                 embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.0):
        super().__init__()
        
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        num_patches = self.patch_embed.n_patches
        
        # Learnable embeddings
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        
        # Transformer blocks
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                dim_feedforward=int(embed_dim * mlp_ratio),
                dropout=0.1
            ),
            num_layers=depth
        )
        
        # Classification head
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
    
    def forward(self, x):
        batch_size = x.shape[0]
        
        # Patch embedding
        x = self.patch_embed(x)
        
        # Add class token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        
        # Add position embedding
        x = x + self.pos_embed
        
        # Transformer
        x = self.transformer(x)
        
        # Classification
        x = self.norm(x[:, 0])  # Use class token
        x = self.head(x)
        
        return x
```

## Training and Optimization

### Data Augmentation
```python
import torchvision.transforms as transforms

# Training transforms
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Validation transforms
val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
```

### Training Loop
```python
def train_model(model, train_loader, val_loader, num_epochs=10):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    best_val_acc = 0.0
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_correct += (predicted == labels).sum().item()
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_correct += (predicted == labels).sum().item()
        
        # Calculate metrics
        train_acc = train_correct / len(train_loader.dataset)
        val_acc = val_correct / len(val_loader.dataset)
        
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'Train Loss: {train_loss/len(train_loader):.4f}, Train Acc: {train_acc:.4f}')
        print(f'Val Loss: {val_loss/len(val_loader):.4f}, Val Acc: {val_acc:.4f}')
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_model.pth')
        
        scheduler.step()
    
    return model
```

## Advanced Techniques

### Transfer Learning
```python
import torchvision.models as models

# Load pre-trained model
model = models.resnet50(pretrained=True)

# Freeze early layers
for param in model.parameters():
    param.requires_grad = False

# Replace final layer
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, num_classes)

# Fine-tune only the final layer initially
optimizer = torch.optim.Adam(model.fc.parameters(), lr=1e-3)
```

### Attention Mechanisms
```python
class SpatialAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # x: (batch_size, channels, height, width)
        attention = self.conv(x)  # (batch_size, 1, height, width)
        attention = self.sigmoid(attention)
        return x * attention

class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction),
            nn.ReLU(),
            nn.Linear(in_channels // reduction, in_channels)
        )
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        batch_size, channels, _, _ = x.size()
        
        # Global average pooling
        avg_out = self.avg_pool(x).view(batch_size, channels)
        avg_out = self.fc(avg_out)
        
        # Global max pooling
        max_out = self.max_pool(x).view(batch_size, channels)
        max_out = self.fc(max_out)
        
        # Combine and apply attention
        attention = self.sigmoid(avg_out + max_out).view(batch_size, channels, 1, 1)
        return x * attention
```

## Model Evaluation and Deployment

### Evaluation Metrics
```python
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def evaluate_model(model, test_loader, class_names):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_predictions, average='weighted'
    )
    
    print(f'Accuracy: {accuracy:.4f}')
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'F1-Score: {f1:.4f}')
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_predictions)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()
```

### Model Optimization
```python
# Model quantization for deployment
def quantize_model(model):
    model.eval()
    quantized_model = torch.quantization.quantize_dynamic(
        model, {nn.Linear, nn.Conv2d}, dtype=torch.qint8
    )
    return quantized_model

# ONNX export for cross-platform deployment
def export_to_onnx(model, input_shape, output_path):
    model.eval()
    dummy_input = torch.randn(1, *input_shape)
    
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output']
    )
```

## Future Directions

### 1. Self-Supervised Learning
- Contrastive learning (SimCLR, MoCo)
- Masked image modeling (MAE, BEiT)
- Multi-modal pre-training (CLIP, ALIGN)

### 2. Efficient Architectures
- MobileNets for mobile deployment
- EfficientNets for optimal accuracy-efficiency trade-offs
- Neural Architecture Search (NAS)

### 3. Multimodal Vision
- Vision-Language models (CLIP, DALL-E)
- Video understanding
- 3D computer vision

## Conclusion

Computer Vision with deep learning has revolutionized how machines perceive and understand visual information. From CNNs to Vision Transformers, each advancement has pushed the boundaries of what's possible.

Key takeaways:
- **CNNs remain fundamental** for many vision tasks
- **Transfer learning** accelerates development and improves performance
- **Vision Transformers** are becoming competitive with CNNs
- **Data augmentation** is crucial for robust models
- **Attention mechanisms** improve model interpretability and performance

The field continues to evolve rapidly, with new architectures and techniques emerging regularly. Stay updated with the latest research and experiment with different approaches to find what works best for your specific use case.