# train_final.py (SIMPLIFIED VERSION - NO MULTIPROCESSING ISSUES)
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, ConcatDataset
import time
import os
import matplotlib.pyplot as plt
from fixed_data_loader import CBISMammoDataset, train_transform, val_transform
from tiny_model import SuperTinyMammoNet

def load_all_data():
    """Load ALL 3103 images"""
    print("="*60)
    print("🚀 LOADING 3,103 IMAGES FOR TRAINING")
    print("="*60)
    
    csv_files = [
        'archive/csv/mass_case_description_train_set.csv',
        'archive/csv/mass_case_description_test_set.csv',
        'archive/csv/calc_case_description_train_set.csv',
        'archive/csv/calc_case_description_test_set.csv'
    ]
    
    datasets = []
    total = 0
    
    for csv_file in csv_files:
        print(f"\n📁 Loading: {csv_file}")
        dataset = CBISMammoDataset(
            csv_path=csv_file,
            jpeg_root='archive/jpeg',
            transform=train_transform,  # Use train_transform for all initially
            max_samples=None
        )
        
        if len(dataset) > 0:
            datasets.append(dataset)
            total += len(dataset)
            print(f"   ✅ Added {len(dataset)} images")
    
    if len(datasets) > 1:
        combined = ConcatDataset(datasets)
        print(f"\n🎯 TOTAL IMAGES: {len(combined)}")
        return combined, datasets  # Return both combined and individual datasets
    else:
        return datasets[0] if datasets else None, datasets

def train_final():
    # Configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n🖥️ Using device: {device}")
    
    # 1. Load ALL 3103 images
    full_dataset, individual_datasets = load_all_data()
    
    if full_dataset is None:
        print("❌ No images loaded!")
        return
    
    # 2. Split into train (80%) and validation (20%)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    # 3. Create data loaders with num_workers=0 to avoid multiprocessing issues
    train_loader = DataLoader(
        train_dataset, 
        batch_size=32, 
        shuffle=True, 
        num_workers=0  # Set to 0 to avoid pickling issues
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=32, 
        shuffle=False, 
        num_workers=0  # Set to 0 to avoid pickling issues
    )
    
    print(f"\n📊 Training samples: {train_size} ({train_size/len(full_dataset)*100:.1f}%)")
    print(f"📊 Validation samples: {val_size} ({val_size/len(full_dataset)*100:.1f}%)")
    
    # 4. Create model
    print("\n🧠 Creating model...")
    model = SuperTinyMammoNet().to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model has {total_params:,} parameters")
    
    # 5. Training settings
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        patience=3, 
        factor=0.5
    )
    
    # 6. Training loop
    epochs = 15  # Reduced to 15 for faster training
    train_losses = []
    val_losses = []
    val_accs = []
    best_val_acc = 0
    
    print("\n🎯 Starting training with 3,103 images...")
    print("="*60)
    start_time = time.time()
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for batch_idx, (images, labels, _) in enumerate(train_loader):
            # Move to device
            images, labels = images.to(device), labels.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Statistics
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
            
            if batch_idx % 10 == 0:
                print(f'Epoch {epoch+1}/{epochs} | Batch {batch_idx}/{len(train_loader)} | Loss: {loss.item():.4f}')
        
        train_acc = 100. * train_correct / train_total
        train_loss_avg = train_loss / len(train_loader)
        
        # Validation phase
        model.eval()
        val_correct = 0
        val_total = 0
        val_loss = 0
        
        with torch.no_grad():
            for images, labels, _ in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
        
        val_acc = 100. * val_correct / val_total
        val_loss_avg = val_loss / len(val_loader)
        
        train_losses.append(train_loss_avg)
        val_losses.append(val_loss_avg)
        val_accs.append(val_acc)
        
        print(f'\n✅ Epoch {epoch+1}: Train Acc = {train_acc:.2f}%, Val Acc = {val_acc:.2f}%')
        print(f'   Train Loss = {train_loss_avg:.4f}, Val Loss = {val_loss_avg:.4f}\n')
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'train_acc': train_acc,
            }, 'models/final_model_3103.pth')
            print(f"💾 SAVED BEST MODEL! Accuracy: {val_acc:.2f}%")
        
        scheduler.step(val_loss_avg)
    
    training_time = time.time() - start_time
    print("="*60)
    print(f"\n✅ TRAINING COMPLETE!")
    print(f"⏱️  Time: {training_time/60:.1f} minutes")
    print(f"🏆 Best validation accuracy: {best_val_acc:.2f}%")
    print(f"📸 Total images used: {len(full_dataset)}")
    
    # Plot training history
    try:
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 3, 1)
        plt.plot(train_losses, label='Train Loss', linewidth=2)
        plt.plot(val_losses, label='Val Loss', linewidth=2)
        plt.title('Loss Over Time')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(1, 3, 2)
        plt.plot(val_accs, label='Validation Accuracy', color='orange', linewidth=2)
        plt.title('Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(1, 3, 3)
        # Count labels
        benign = 0
        malignant = 0
        for ds in individual_datasets:
            benign += ds.labels.count(0)
            malignant += ds.labels.count(1)
        
        plt.bar(['Benign', 'Malignant'], [benign, malignant], 
                color=['green', 'red'])
        plt.title(f'Dataset Distribution (Total: {len(full_dataset)})')
        plt.ylabel('Number of Images')
        
        plt.tight_layout()
        plt.savefig('models/training_summary_3103.png', dpi=150)
        plt.show()
    except Exception as e:
        print(f"Note: Plotting failed but model saved: {e}")
    
    return model

def quick_test(model, device):
    """Quick test with sample images"""
    print("\n🔍 Quick test on sample images...")
    model.eval()
    
    # Load a few test images
    test_dataset = CBISMammoDataset(
        csv_path='archive/csv/mass_case_description_test_set.csv',
        jpeg_root='archive/jpeg',
        transform=val_transform,
        max_samples=20
    )
    
    correct = 0
    total = min(10, len(test_dataset))
    
    with torch.no_grad():
        for i in range(total):
            img, label, path = test_dataset[i]
            img = img.unsqueeze(0).to(device)
            output = model(img)
            pred = output.argmax().item()
            prob = torch.softmax(output, dim=1)[0][pred].item()
            
            result = "✓" if pred == label else "✗"
            print(f"{result} {os.path.basename(path)}")
            print(f"   True: {'MALIGNANT' if label else 'BENIGN'}")
            print(f"   Pred: {'MALIGNANT' if pred else 'BENIGN'} ({prob:.1%})")
            if pred == label:
                correct += 1
    
    print(f"\n📊 Quick test accuracy: {correct}/{total} = {correct/total*100:.1f}%")

if __name__ == "__main__":
    # Create models directory
    os.makedirs('models', exist_ok=True)
    
    # Train the model
    model = train_final()
    
    if model:
        # Quick test
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        quick_test(model, device)
        
        print("\n" + "="*60)
        print("🎉 FINAL MODEL READY!")
        print("="*60)
        print("\nFiles saved:")
        print("  📁 models/final_model_3103.pth - Trained model")
        if os.path.exists('models/training_summary_3103.png'):
            print("  📁 models/training_summary_3103.png - Training plots")