# fixed_data_loader.py (ENHANCED VERSION - REPLACE YOUR CURRENT FILE)
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, ConcatDataset
from PIL import Image
import torch
import torchvision.transforms as transforms
import os
import glob
import re

class CBISMammoDataset(Dataset):
    def __init__(self, csv_path, jpeg_root, transform=None, max_samples=None):
        """
        Enhanced dataset loader for CBIS-DDSM format with multiple matching strategies
        """
        # Load the CSV
        self.df = pd.read_csv(csv_path)
        self.jpeg_root = jpeg_root
        self.transform = transform
        
        # Create binary labels (1 for malignant, 0 for benign/normal)
        if 'pathology' in self.df.columns:
            self.df['label'] = (self.df['pathology'] == 'MALIGNANT').astype(int)
        else:
            print(f"Warning: 'pathology' column not found in {csv_path}")
            self.df['label'] = 0
        
        # Get all JPEG files and create lookup
        print(f"Scanning for JPEG files in {jpeg_root}...")
        all_jpegs = glob.glob(os.path.join(jpeg_root, '**', '*.jpg'), recursive=True)
        print(f"Found {len(all_jpegs)} JPEG files")
        
        # Create multiple lookup strategies
        self.folder_to_jpegs = {}  # Full folder name -> images
        self.id_to_jpegs = {}       # Numeric ID -> images
        self.patient_to_jpegs = {}  # Patient ID -> images
        self.filename_to_jpegs = {} # Filename -> images (for direct matching)
        
        for path in all_jpegs:
            folder_name = os.path.basename(os.path.dirname(path))
            filename = os.path.basename(path)
            
            # Store by folder
            if folder_name not in self.folder_to_jpegs:
                self.folder_to_jpegs[folder_name] = []
            self.folder_to_jpegs[folder_name].append(path)
            
            # Store by filename
            if filename not in self.filename_to_jpegs:
                self.filename_to_jpegs[filename] = []
            self.filename_to_jpegs[filename].append(path)
            
            # Extract numeric ID from folder
            id_match = re.search(r'(1\.3\.6\.1\.4\.1\.9590\.100\.1\.2\.\d+)', folder_name)
            if id_match:
                num_id = id_match.group(1)
                if num_id not in self.id_to_jpegs:
                    self.id_to_jpegs[num_id] = []
                self.id_to_jpegs[num_id].append(path)
            
            # Extract patient ID if present
            patient_match = re.search(r'P_(\d+)', folder_name)
            if patient_match:
                pat_id = patient_match.group(1)
                if pat_id not in self.patient_to_jpegs:
                    self.patient_to_jpegs[pat_id] = []
                self.patient_to_jpegs[pat_id].append(path)
        
        print(f"Found {len(self.folder_to_jpegs)} folders")
        print(f"Found {len(self.id_to_jpegs)} numeric IDs")
        print(f"Found {len(self.patient_to_jpegs)} patient IDs")
        
        # Find the image path column
        self.img_col = None
        for col in self.df.columns:
            if 'image file path' in col.lower():
                self.img_col = col
                break
        
        if not self.img_col:
            for col in self.df.columns:
                if 'file path' in col.lower():
                    self.img_col = col
                    break
        
        if not self.img_col:
            # Try any column that might contain paths
            for col in self.df.columns:
                if self.df[col].dtype == 'object' and len(self.df) > 0:
                    sample = str(self.df[col].iloc[0])
                    if 'Mass' in sample or 'Calc' in sample or 'P_' in sample:
                        self.img_col = col
                        break
        
        print(f"Using image column: {self.img_col}")
        
        # Match images using multiple strategies
        self.image_paths = []
        self.labels = []
        matched_count = 0
        self.strategy_counts = {'folder': 0, 'numeric_id': 0, 'patient_id': 0, 'filename': 0, 'partial': 0}
        
        for idx, row in self.df.iterrows():
            if max_samples and len(self.image_paths) >= max_samples:
                break
            
            if self.img_col and pd.notna(row[self.img_col]):
                csv_path_val = str(row[self.img_col])
                found = False
                
                # Strategy 1: Extract numeric IDs from CSV path
                numeric_ids = re.findall(r'(1\.3\.6\.1\.4\.1\.9590\.100\.1\.2\.\d+)', csv_path_val)
                
                # Try each numeric ID
                for num_id in numeric_ids:
                    if num_id in self.id_to_jpegs:
                        for jpeg_path in self.id_to_jpegs[num_id]:
                            self.image_paths.append(jpeg_path)
                            self.labels.append(row['label'])
                            matched_count += 1
                            self.strategy_counts['numeric_id'] += 1
                            found = True
                        if found:
                            break
                
                # Strategy 2: Extract patient ID
                if not found:
                    patient_match = re.search(r'P_(\d+)', csv_path_val)
                    if patient_match:
                        pat_id = patient_match.group(1)
                        if pat_id in self.patient_to_jpegs:
                            for jpeg_path in self.patient_to_jpegs[pat_id]:
                                self.image_paths.append(jpeg_path)
                                self.labels.append(row['label'])
                                matched_count += 1
                                self.strategy_counts['patient_id'] += 1
                                found = True
                
                # Strategy 3: Look at folder parts
                if not found:
                    parts = csv_path_val.split('/')
                    for part in parts:
                        if part in self.folder_to_jpegs:
                            for jpeg_path in self.folder_to_jpegs[part]:
                                self.image_paths.append(jpeg_path)
                                self.labels.append(row['label'])
                                matched_count += 1
                                self.strategy_counts['folder'] += 1
                                found = True
                            break
                
                # Strategy 4: Try filename matching
                if not found:
                    filename = os.path.basename(csv_path_val).replace('.dcm', '.jpg')
                    if filename in self.filename_to_jpegs:
                        for jpeg_path in self.filename_to_jpegs[filename]:
                            self.image_paths.append(jpeg_path)
                            self.labels.append(row['label'])
                            matched_count += 1
                            self.strategy_counts['filename'] += 1
                            found = True
                
                # Strategy 5: Try partial matching (first 50 chars of IDs)
                if not found and numeric_ids:
                    for num_id in numeric_ids:
                        id_prefix = num_id[:50]
                        for folder_id in self.id_to_jpegs.keys():
                            if folder_id.startswith(id_prefix):
                                for jpeg_path in self.id_to_jpegs[folder_id]:
                                    self.image_paths.append(jpeg_path)
                                    self.labels.append(row['label'])
                                    matched_count += 1
                                    self.strategy_counts['partial'] += 1
                                    found = True
                                break
                        if found:
                            break
            
            # Progress update
            if matched_count > 0 and matched_count % 100 == 0:
                print(f"  Matched {matched_count} images...")
        
        # Remove duplicates
        unique_paths = []
        unique_labels = []
        seen = set()
        for path, label in zip(self.image_paths, self.labels):
            if path not in seen:
                seen.add(path)
                unique_paths.append(path)
                unique_labels.append(label)
        
        self.image_paths = unique_paths
        self.labels = unique_labels
        
        print(f"\n✅ Successfully loaded {len(self.image_paths)} unique images")
        print(f"   Matching strategies: {self.strategy_counts}")
        if len(self.image_paths) > 0:
            benign = len([l for l in self.labels if l == 0])
            malignant = len([l for l in self.labels if l == 1])
            print(f"   - Benign/Normal: {benign}")
            print(f"   - Malignant: {malignant}")
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            image = Image.new('RGB', (224, 224), color='black')
        
        if self.transform:
            image = self.transform(image)
            
        return image, label, img_path

# Simple transforms with data augmentation for training
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.3),
    transforms.RandomRotation(5),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                        std=[0.229, 0.224, 0.225])
])

# Validation transform (no augmentation)
val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                        std=[0.229, 0.224, 0.225])
])

# For backward compatibility
simple_transform = train_transform

def test_loader():
    """Test function to verify the loader works"""
    print("="*60)
    print("TESTING ENHANCED DATA LOADER")
    print("="*60)
    
    # Test with mass training set
    dataset = CBISMammoDataset(
        csv_path='archive/csv/mass_case_description_train_set.csv',
        jpeg_root='archive/jpeg',
        transform=None,
        max_samples=200
    )
    
    if len(dataset) > 0:
        print(f"\n✅ SUCCESS! Loaded {len(dataset)} images")
        
        # Show a few samples
        print("\n📸 Sample images:")
        for i in range(min(3, len(dataset))):
            img, label, path = dataset[i]
            print(f"\nImage {i+1}:")
            print(f"  File: {os.path.basename(path)}")
            print(f"  Label: {'MALIGNANT' if label == 1 else 'BENIGN'}")
        
        return True
    else:
        print("\n❌ No images loaded")
        return False

if __name__ == "__main__":
    test_loader()