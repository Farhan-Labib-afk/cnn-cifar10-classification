"""
CNN Image Classification on CIFAR-10
Project for RBC Borealis ML Co-op Application
Author: Farhan Labib
Date: December 2025

This project demonstrates:
- Custom CNN architecture from scratch
- Transfer learning with pre-trained ResNet18
- Model evaluation and comparison
- Visualization of results
"""

from pathlib import Path
import json

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
from tqdm import tqdm
import time

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

BASE_DIR = Path(__file__).resolve().parent
ASSETS_DIR = BASE_DIR / "assets"
MODELS_DIR = BASE_DIR / "models"

# ============================================================================
# PART 1: DATA LOADING AND PREPROCESSING
# ============================================================================

def load_cifar10_data(batch_size=128):
    """
    Load CIFAR-10 dataset with appropriate transforms.
    
    CIFAR-10 contains 60,000 32x32 color images in 10 classes:
    - 50,000 training images
    - 10,000 test images
    """
    
    # Data augmentation for training (improves generalization)
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), 
                           (0.2023, 0.1994, 0.2010))
    ])
    
    # No augmentation for test set
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), 
                           (0.2023, 0.1994, 0.2010))
    ])
    
    # Download and load datasets
    train_dataset = torchvision.datasets.CIFAR10(
        root=BASE_DIR / 'data', train=True, download=True, transform=train_transform
    )
    
    test_dataset = torchvision.datasets.CIFAR10(
        root=BASE_DIR / 'data', train=False, download=True, transform=test_transform
    )
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                            shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, 
                           shuffle=False, num_workers=2)
    
    return train_loader, test_loader

# Class names for CIFAR-10
classes = ('plane', 'car', 'bird', 'cat', 'deer', 
           'dog', 'frog', 'horse', 'ship', 'truck')

# ============================================================================
# PART 2: CUSTOM CNN ARCHITECTURE
# ============================================================================

class CustomCNN(nn.Module):
    """
    Custom CNN architecture for CIFAR-10 classification.
    
    Architecture:
    - 3 Convolutional blocks with batch normalization and max pooling
    - 2 Fully connected layers with dropout
    - ~1.2M parameters
    """
    
    def __init__(self, num_classes=10):
        super(CustomCNN, self).__init__()
        
        # Convolutional Block 1
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)  # 32x32 -> 16x16
        )
        
        # Convolutional Block 2
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)  # 16x16 -> 8x8
        )
        
        # Convolutional Block 3
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)  # 8x8 -> 4x4
        )
        
        # Fully Connected Layers
        self.fc = nn.Sequential(
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc(x)
        return x

# ============================================================================
# PART 3: TRANSFER LEARNING WITH RESNET18
# ============================================================================

def create_resnet18_transfer(num_classes=10):
    """
    Create ResNet18 model with transfer learning.
    
    Pre-trained on ImageNet, fine-tuned for CIFAR-10.
    """
    try:
        weights = torchvision.models.ResNet18_Weights.DEFAULT
        model = torchvision.models.resnet18(weights=weights)
    except AttributeError:
        model = torchvision.models.resnet18(pretrained=True)
    
    # Freeze early layers (optional - uncomment to freeze)
    # for param in model.parameters():
    #     param.requires_grad = False
    
    # Replace final fully connected layer
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)
    
    return model

# ============================================================================
# PART 4: TRAINING FUNCTION
# ============================================================================

def train_model(model, train_loader, test_loader, num_epochs=20, 
                learning_rate=0.001, model_name="model"):
    """
    Train a model and track performance metrics.
    """
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', 
                                                     factor=0.5, patience=3)
    
    model = model.to(device)
    
    # Track metrics
    train_losses = []
    train_accs = []
    test_accs = []
    best_acc = 0.0
    
    print(f"\n{'='*60}")
    print(f"Training {model_name}")
    print(f"{'='*60}\n")
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for inputs, labels in pbar:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Statistics
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{running_loss/len(train_loader):.3f}',
                'acc': f'{100.*correct/total:.2f}%'
            })
        
        train_loss = running_loss / len(train_loader)
        train_acc = 100. * correct / total
        
        # Evaluation phase
        test_acc = evaluate_model(model, test_loader)
        
        # Track metrics
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        test_accs.append(test_acc)
        
        # Learning rate scheduling
        scheduler.step(test_acc)
        
        # Save best model
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), MODELS_DIR / f'{model_name}_best.pth')
        
        print(f"Epoch {epoch+1}: Train Loss: {train_loss:.3f}, "
              f"Train Acc: {train_acc:.2f}%, Test Acc: {test_acc:.2f}%")
    
    print(f"\nBest Test Accuracy: {best_acc:.2f}%\n")
    
    return {
        'train_losses': train_losses,
        'train_accs': train_accs,
        'test_accs': test_accs,
        'best_acc': best_acc
    }

# ============================================================================
# PART 5: EVALUATION FUNCTION
# ============================================================================

def evaluate_model(model, test_loader):
    """
    Evaluate model on test set.
    """
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    accuracy = 100. * correct / total
    return accuracy

def detailed_evaluation(model, test_loader):
    """
    Perform detailed evaluation with confusion matrix and classification report.
    """
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=classes))
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    return all_preds, all_labels, cm

# ============================================================================
# PART 6: VISUALIZATION FUNCTIONS
# ============================================================================

def plot_training_history(history_custom, history_resnet):
    """
    Plot training curves comparing both models.
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot 1: Training Loss
    axes[0].plot(history_custom['train_losses'], label='Custom CNN', marker='o')
    axes[0].plot(history_resnet['train_losses'], label='ResNet18', marker='s')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Training Loss')
    axes[0].set_title('Training Loss Comparison')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Test Accuracy
    axes[1].plot(history_custom['test_accs'], label='Custom CNN', marker='o')
    axes[1].plot(history_resnet['test_accs'], label='ResNet18', marker='s')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Test Accuracy (%)')
    axes[1].set_title('Test Accuracy Comparison')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(ASSETS_DIR / 'training_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Training curves saved in assets/")

def plot_confusion_matrix(cm, title='Confusion Matrix'):
    """
    Plot confusion matrix heatmap.
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=classes, yticklabels=classes)
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(ASSETS_DIR / f'{title.lower().replace(" ", "_")}.png', dpi=300, bbox_inches='tight')
    plt.show()

def visualize_predictions(model, test_loader, num_images=10):
    """
    Visualize model predictions on sample images.
    """
    model.eval()
    
    # Get one batch
    dataiter = iter(test_loader)
    images, labels = next(dataiter)
    images, labels = images.to(device), labels
    
    # Get predictions
    with torch.no_grad():
        outputs = model(images)
        _, predicted = outputs.max(1)
        predicted = predicted.cpu()
    
    # Plot
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    axes = axes.ravel()
    
    for i in range(num_images):
        img = images[i].cpu()
        
        # Denormalize
        img = img * torch.tensor([0.2023, 0.1994, 0.2010]).view(3, 1, 1)
        img = img + torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1)
        img = torch.clamp(img, 0, 1)
        
        # Plot
        axes[i].imshow(img.permute(1, 2, 0))
        axes[i].set_title(f'True: {classes[labels[i]]}\n'
                         f'Pred: {classes[predicted[i]]}',
                         color='green' if labels[i] == predicted[i] else 'red')
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig(ASSETS_DIR / 'sample_predictions.png', dpi=300, bbox_inches='tight')
    plt.show()

# ============================================================================
# PART 7: MAIN EXECUTION
# ============================================================================

def main():
    """
    Main execution function.
    """
    print("="*60)
    print("CNN Image Classification on CIFAR-10")
    print("="*60)

    ASSETS_DIR.mkdir(exist_ok=True)
    MODELS_DIR.mkdir(exist_ok=True)
    
    # Load data
    print("\n[1/6] Loading CIFAR-10 dataset...")
    train_loader, test_loader = load_cifar10_data(batch_size=128)
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")
    
    # Model 1: Custom CNN
    print("\n[2/6] Creating Custom CNN architecture...")
    model_custom = CustomCNN(num_classes=10)
    num_params_custom = sum(p.numel() for p in model_custom.parameters())
    print(f"Custom CNN parameters: {num_params_custom:,}")
    
    # Model 2: ResNet18 Transfer Learning
    print("\n[3/6] Creating ResNet18 with transfer learning...")
    model_resnet = create_resnet18_transfer(num_classes=10)
    num_params_resnet = sum(p.numel() for p in model_resnet.parameters())
    print(f"ResNet18 parameters: {num_params_resnet:,}")
    
    # Train Custom CNN
    print("\n[4/6] Training Custom CNN...")
    start_time = time.time()
    history_custom = train_model(model_custom, train_loader, test_loader, 
                                 num_epochs=20, learning_rate=0.001,
                                 model_name="custom_cnn")
    time_custom = time.time() - start_time
    print(f"Training time: {time_custom/60:.2f} minutes")
    
    # Train ResNet18
    print("\n[5/6] Training ResNet18 with transfer learning...")
    start_time = time.time()
    history_resnet = train_model(model_resnet, train_loader, test_loader, 
                                 num_epochs=15, learning_rate=0.0001,
                                 model_name="resnet18_transfer")
    time_resnet = time.time() - start_time
    print(f"Training time: {time_resnet/60:.2f} minutes")
    
    # Visualizations and final evaluation
    print("\n[6/6] Generating visualizations and final evaluation...")
    
    # Plot training history
    plot_training_history(history_custom, history_resnet)
    
    # Load best models for final evaluation
    model_custom.load_state_dict(torch.load(MODELS_DIR / 'custom_cnn_best.pth', map_location=device))
    model_resnet.load_state_dict(torch.load(MODELS_DIR / 'resnet18_transfer_best.pth', map_location=device))
    model_custom = model_custom.to(device)
    model_resnet = model_resnet.to(device)
    
    # Detailed evaluation - Custom CNN
    print("\n" + "="*60)
    print("CUSTOM CNN - Detailed Evaluation")
    print("="*60)
    preds_custom, labels_custom, cm_custom = detailed_evaluation(
        model_custom, test_loader
    )
    plot_confusion_matrix(cm_custom, title='Custom CNN Confusion Matrix')
    
    # Detailed evaluation - ResNet18
    print("\n" + "="*60)
    print("RESNET18 - Detailed Evaluation")
    print("="*60)
    preds_resnet, labels_resnet, cm_resnet = detailed_evaluation(
        model_resnet, test_loader
    )
    plot_confusion_matrix(cm_resnet, title='ResNet18 Confusion Matrix')
    
    # Visualize predictions
    print("\nGenerating sample predictions...")
    visualize_predictions(model_resnet, test_loader)
    
    # Final summary
    print("\n" + "="*60)
    print("FINAL RESULTS SUMMARY")
    print("="*60)
    print(f"\nCustom CNN:")
    print(f"  - Parameters: {num_params_custom:,}")
    print(f"  - Best Test Accuracy: {history_custom['best_acc']:.2f}%")
    print(f"  - Training Time: {time_custom/60:.2f} minutes")
    
    print(f"\nResNet18 Transfer Learning:")
    print(f"  - Parameters: {num_params_resnet:,}")
    print(f"  - Best Test Accuracy: {history_resnet['best_acc']:.2f}%")
    print(f"  - Training Time: {time_resnet/60:.2f} minutes")
    
    print(f"\nPerformance Gain: {history_resnet['best_acc'] - history_custom['best_acc']:.2f}%")
    print(f"Efficiency: ResNet18 is {num_params_resnet/num_params_custom:.1f}x larger")

    results_path = BASE_DIR / "results.json"
    results = {
        "custom_best_acc": f"{history_custom['best_acc']:.2f}%",
        "custom_time_min": f"{time_custom/60:.2f}",
        "resnet_best_acc": f"{history_resnet['best_acc']:.2f}%",
        "resnet_time_min": f"{time_resnet/60:.2f}",
    }
    results_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Results saved to {results_path}")
    
    print("\n" + "="*60)
    print("Project completed! All visualizations saved.")
    print("="*60)

# ============================================================================
# RUN THE PROJECT
# ============================================================================

if __name__ == "__main__":
    main()
