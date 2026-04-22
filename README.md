# AI-Powered Mammogram Analysis System

An **explainable AI system for breast cancer detection from mammogram images.** Built with a lightweight CNN (98K parameters) and Grad-CAM for interpretable predictions. Designed for edge deployment, can run on any laptop without GPU.

## Overview

Breast cancer is the leading cause of cancer death among women worldwide. Early detection increases survival rates to over 90%, yet current screening methods face critical limitations:
- **Missed cancers**: Radiologists miss up to 30% of visible cancers due to fatigue
- **Specialist shortage**: Many regions lack access to trained radiologists  
- **Delayed diagnosis**: Patients wait weeks for results

**Our solution**: An AI-powered second opinion system that:
- Detects suspicious regions in mammograms
- Provides a risk prediction score
- Generates heatmap explanations using Grad-CAM
- Works on any laptop (no GPU required)

## Features

| Feature | Description |
|---------|-------------|
| **Lightweight CNN** | 98K parameters, 0.4MB model size - 250x smaller than ResNet50 |
| **Grad-CAM Explainability** | Heatmaps showing which regions influenced predictions |
| **Streamlit Web App** | User-friendly interface for real-time analysis |
| **Fast Inference** | <0.1 seconds per image on CPU |
| **Clinically Focused** | Built on Kaggle's CBIS-DDSM dataset |

## Dataset

**Kaggle's CBIS-DDSM (Curated Breast Imaging Subset of DDSM)**
- Total JPEG images: 10,237
- Successfully matched: 3,103 images (87% match rate)
- Class distribution: 56% Benign, 44% Malignant

## Architecture
SuperTinyMammoNet (98,178 parameters)
├── Block 1: Conv2d(3→16) + BatchNorm + ReLU + MaxPool
├── Block 2: Conv2d(16→32) + BatchNorm + ReLU + MaxPool
├── Block 3: Conv2d(32→64) + BatchNorm + ReLU + MaxPool
├── Block 4: Conv2d(64→128) + BatchNorm + ReLU + MaxPool
└── Classifier: AdaptiveAvgPool + Flatten + Dropout(0.3) + Linear(128→2)

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager
- Git (optional,for cloning)

### Step-by-Step Setup

```bash
# Clone the repository
git clone https://github.com/codingwithdc/AI-powered-mammogram-analysis-system.git
cd AI-powered-mammogram-analysis-system

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download the CBIS-DDSM dataset from Kaggle from here: https://www.kaggle.com/datasets/awsaf49/cbis-ddsm-breast-cancer-image-dataset
and place the archive folder in the project root.

Expected folder structure:

text
AI-powered-mammogram-analysis-system/
├── archive/
│   ├── csv/
│   │   ├── mass_case_description_train_set.csv
│   │   ├── mass_case_description_test_set.csv
│   │   ├── calc_case_description_train_set.csv
│   │   └── calc_case_description_test_set.csv
│   └── jpeg/
│       └── (10,237 mammogram images)
├── app.py
├── train_final.py
└── ...

#Train the Model
bash
python train_final.py
The trained model (final_model_3103.pth) will be saved in the models directory.

#Run the Web App
bash
streamlit run app.py

Then open http://localhost:8501 in your browser.


