"""
U-Net Training Script for Floor Plan Cleaning
===============================================
Trains the U-Net model to convert noisy SLAM occupancy grids
into clean, semantically segmented floor plans.

Classes:
    0 = Background
    1 = Walkable Space (rooms, corridors)
    2 = Wall

Usage:
    python train.py
    
    This will train for 15 epochs on 2000 synthetic samples
    and save the best model weights to 'best_model.pth'.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import SyntheticFloorPlanDataset
from model import UNet
import time
import os


def compute_iou(pred, target, num_classes=3):
    """Compute mean Intersection-over-Union across all classes."""
    ious = []
    pred = pred.view(-1)
    target = target.view(-1)
    
    for cls in range(num_classes):
        pred_cls = (pred == cls)
        target_cls = (target == cls)
        intersection = (pred_cls & target_cls).sum().float()
        union = (pred_cls | target_cls).sum().float()
        if union == 0:
            ious.append(1.0)
        else:
            ious.append((intersection / union).item())
    
    return sum(ious) / len(ious)


def train():
    # =====================
    # Hyperparameters
    # =====================
    BATCH_SIZE = 4
    EPOCHS = 10
    LEARNING_RATE = 1e-3
    TRAIN_SAMPLES = 200
    VAL_SAMPLES = 50
    IMAGE_SIZE = 256
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("=" * 60)
    print("  U-Net Floor Plan Cleaning Model -- Training")
    print("=" * 60)
    print(f"  Device:          {device}")
    print(f"  Batch Size:      {BATCH_SIZE}")
    print(f"  Epochs:          {EPOCHS}")
    print(f"  Train Samples:   {TRAIN_SAMPLES}")
    print(f"  Val Samples:     {VAL_SAMPLES}")
    print(f"  Image Size:      {IMAGE_SIZE}x{IMAGE_SIZE}")
    print(f"  Learning Rate:   {LEARNING_RATE}")
    print("=" * 60)
    print()

    # =====================
    # Data
    # =====================
    print("Initializing datasets...")
    train_dataset = SyntheticFloorPlanDataset(size=IMAGE_SIZE, length=TRAIN_SAMPLES, seed=42)
    val_dataset = SyntheticFloorPlanDataset(size=IMAGE_SIZE, length=VAL_SAMPLES, seed=123)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches:   {len(val_loader)}")
    print()

    # =====================
    # Model
    # =====================
    model = UNet(n_channels=1, n_classes=3).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model loaded: {total_params:,} parameters")
    print()
    
    # Weighted loss: walls (class 2) are thin and rare, so upweight them
    class_weights = torch.tensor([1.0, 2.0, 10.0]).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Learning rate scheduler: reduce on plateau
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3
    )

    # =====================
    # Training Loop
    # =====================
    best_val_loss = float('inf')
    best_val_iou = 0.0
    
    for epoch in range(EPOCHS):
        epoch_start = time.time()
        
        # --- Train ---
        model.train()
        train_loss = 0
        for i, (images, masks) in enumerate(train_loader):
            images = images.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            
            if (i + 1) % 10 == 0:
                print(f"  Epoch [{epoch+1}/{EPOCHS}] Step [{i+1}/{len(train_loader)}] Loss: {loss.item():.4f}")

        avg_train_loss = train_loss / len(train_loader)
        
        # --- Validate ---
        model.eval()
        val_loss = 0
        val_iou = 0
        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(device)
                masks = masks.to(device)
                outputs = model(images)
                loss = criterion(outputs, masks)
                val_loss += loss.item()
                
                preds = torch.argmax(outputs, dim=1)
                val_iou += compute_iou(preds, masks)
                
        avg_val_loss = val_loss / len(val_loader)
        avg_val_iou = val_iou / len(val_loader)
        
        epoch_time = time.time() - epoch_start
        
        print()
        print(f"  Epoch [{epoch+1}/{EPOCHS}] -- {epoch_time:.1f}s")
        print(f"    Train Loss: {avg_train_loss:.4f}")
        print(f"    Val Loss:   {avg_val_loss:.4f}")
        print(f"    Val mIoU:   {avg_val_iou:.4f}")
        
        # Step scheduler
        scheduler.step(avg_val_loss)
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_val_iou = avg_val_iou
            torch.save(model.state_dict(), 'best_model.pth')
            print(f"    [OK] New best model saved! (Val Loss: {best_val_loss:.4f}, mIoU: {best_val_iou:.4f})")
        
        print()

    print("=" * 60)
    print("  Training Complete!")
    print(f"  Best Val Loss: {best_val_loss:.4f}")
    print(f"  Best Val mIoU: {best_val_iou:.4f}")
    print("  Model saved to: best_model.pth")
    print("=" * 60)


if __name__ == '__main__':
    train()
