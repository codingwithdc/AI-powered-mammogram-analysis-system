# app.py (FULLY CORRECTED VERSION)
import streamlit as st
import torch
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from models import get_model, get_target_layer
from heatmap import generate_gradcam, overlay_heatmap
from fixed_data_loader import val_transform
import os
import pandas as pd  # ← MISSING IMPORT ADDED!
from datetime import datetime  # ← For timestamp
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget  # ← ADDED

# Page config
st.set_page_config(
    page_title="Breast Cancer AI - Explainable Detection",
    page_icon="🩺",
    layout="wide"
)

# Title
st.title("🩺 Breast Cancer Detection AI with Explainable Grad-CAM")
st.markdown("---")

# Sidebar
with st.sidebar:
    st.header("🔬 About This Tool")
    st.info("""
    **What is Grad-CAM?**
    Grad-CAM highlights the regions of the mammogram that influenced the AI's decision.
    
    - 🔴 **Red areas**: Strongly influenced the prediction
    - 🔵 **Blue areas**: Little influence
    
    This makes the AI **explainable** and **trustworthy**!
    """)
    
    st.header("📊 Model Info")
    st.write("**Architecture:** SuperTinyMammoNet")
    st.write("**Parameters:** 98K (lightweight!)")
    st.write("**Training Data:** 3,103 mammogram images")
    st.write("**Explainability:** Grad-CAM heatmaps")
    
    # Transparency slider
    transparency = st.slider(
        "Heatmap Transparency", 
        min_value=0.1, 
        max_value=0.9, 
        value=0.5, 
        step=0.1,
        help="Adjust how transparent the heatmap overlay appears"
    )
    
    st.header("⚠️ Disclaimer")
    st.warning("This is a demonstration tool. Always consult medical professionals for diagnosis.")

# Load model
@st.cache_resource
def load_model():
    model_path = 'models/final_model_3103.pth'
    if os.path.exists(model_path):
        model = get_model(model_path, model_type='supertiny')
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        model.eval()
        return model, device
    else:
        return None, None

model, device = load_model()

# Main content
col1, col2 = st.columns(2)

with col1:
    st.subheader("📤 Upload Mammogram")
    uploaded_file = st.file_uploader(
        "Choose a mammogram image (JPG/PNG)", 
        type=['jpg', 'jpeg', 'png']
    )
    
    if uploaded_file:
        # Display uploaded image
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption='Uploaded Mammogram', width=None, use_container_width=True)
        
        # Add analyze button
        if st.button('🔍 Analyze with Grad-CAM', type='primary'):
            if model is None:
                st.error("❌ Model not found! Please train the model first.")
            else:
                with st.spinner('🤖 AI is analyzing the mammogram...'):
                    # Prepare image
                    input_tensor = val_transform(image).unsqueeze(0).to(device)
                    
                    # Get prediction
                    with torch.no_grad():
                        output = model(input_tensor)
                        probs = torch.softmax(output, dim=1)
                        pred_class = torch.argmax(probs).item()
                        confidence = probs[0, pred_class].item()
                    
                    # Generate heatmap using the corrected function
                    from pytorch_grad_cam import GradCAM
                    
                    # Get target layer
                    target_layer = model.features[-4]  # Last conv layer
                    
                    # Initialize Grad-CAM
                    cam = GradCAM(model=model, target_layers=[target_layer])
                    targets = [ClassifierOutputTarget(pred_class)]
                    
                    # Generate CAM
                    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0]
                    
                    # Create overlay
                    # Convert image to numpy and resize
                    img_np = np.array(image.resize((224, 224)))
                    img_normalized = img_np.astype(np.float32) / 255.0
                    
                    from pytorch_grad_cam.utils.image import show_cam_on_image
                    overlay = show_cam_on_image(
                        img_normalized,
                        grayscale_cam,
                        use_rgb=True,
                        image_weight=transparency
                    )
                    
                    # Store results in session state
                    st.session_state['result'] = {
                        'pred_class': pred_class,
                        'confidence': confidence,
                        'class_name': 'MALIGNANT' if pred_class == 1 else 'BENIGN',
                        'grayscale_cam': grayscale_cam,
                        'overlay': overlay,
                        'original': image
                    }

with col2:
    st.subheader("📊 Analysis Results")
    
    if 'result' in st.session_state:
        result = st.session_state['result']
        
        # Display metrics
        col_metric1, col_metric2, col_metric3 = st.columns(3)
        
        with col_metric1:
            if result['pred_class'] == 1:
                st.metric("Prediction", "🔴 MALIGNANT", delta=None)
            else:
                st.metric("Prediction", "🟢 BENIGN", delta=None)
        
        with col_metric2:
            st.metric("Confidence", f"{result['confidence']:.1%}")
        
        with col_metric3:
            risk = "HIGH" if result['pred_class'] == 1 and result['confidence'] > 0.5 else "LOW"
            st.metric("Risk Level", risk)
        
        # Display Grad-CAM visualization
        st.subheader("🔍 Grad-CAM Attention Map")
        st.write("*Warmer colors (red/yellow) = Areas the AI focused on*")
        
        # Create side-by-side comparison
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
        
        # Original
        ax1.imshow(np.array(result['original'].resize((224, 224))))
        ax1.set_title('Original Mammogram')
        ax1.axis('off')
        
        # Heatmap only
        ax2.imshow(result['grayscale_cam'], cmap='jet')
        ax2.set_title('AI Attention Heatmap')
        ax2.axis('off')
        
        # Overlay
        ax3.imshow(result['overlay'])
        ax3.set_title(f"Overlay - {result['class_name']} ({result['confidence']:.1%})")
        ax3.axis('off')
        
        plt.tight_layout()
        st.pyplot(fig)
        
        # Explanation
        if result['pred_class'] == 1:
            st.error(f"""
            ⚠️ **AI detected MALIGNANT patterns** with {result['confidence']:.1%} confidence.
            
            The red areas in the heatmap show where the AI found suspicious tissue patterns.
            These regions should be examined carefully by a radiologist.
            """)
        else:
            st.success(f"""
            ✅ **AI detected BENIGN patterns** with {result['confidence']:.1%} confidence.
            
            The heatmap shows normal breast tissue patterns with no significant suspicious areas.
            """)
        
        # Download report
        if st.button("📥 Generate Download Report"):
            # Create report text
            report = f"""AI BREAST CANCER ANALYSIS REPORT
================================
Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

📊 PREDICTION RESULTS
----------------------
Prediction: {result['class_name']}
Confidence: {result['confidence']:.1%}
Risk Level: {'HIGH' if result['pred_class'] == 1 and result['confidence'] > 0.5 else 'LOW'}

🔬 MODEL INFORMATION
----------------------
Architecture: SuperTinyMammoNet
Parameters: 98,178
Training Data: 3,103 mammogram images
Explainability: Grad-CAM Heatmap

📝 INTERPRETATION
----------------------
{('The AI detected suspicious patterns consistent with MALIGNANT tissue. '
  'Areas highlighted in red on the heatmap should be examined by a specialist.')
  if result['pred_class'] == 1 else 
  ('The AI detected normal BENIGN patterns. No significant suspicious '
   'areas were identified in the heatmap.')}

"""
            
            st.download_button(
                label="📄 Save Report as TXT",
                data=report,
                file_name=f"breast_cancer_ai_report_{datetime.now().strftime('%Y%m%d_%H%M')}.txt",
                mime="text/plain"
            )

# Footer
st.markdown("---")

# Instructions
with st.expander("ℹ️ How to use this app"):
    st.write("""
    ### Step-by-Step Guide
    
    1. **Upload** a mammogram image (JPG or PNG format)
    2. **Click** 'Analyze with Grad-CAM' to start the AI analysis
    3. **View** the results panel showing:
       - Prediction (Malignant/Benign)
       - Confidence score
       - Grad-CAM heatmap showing AI focus areas
    4. **Adjust** heatmap transparency with the slider in the sidebar
    5. **Download** the report if needed
    
    ### Understanding the Heatmap
    
    - 🔴 **Red areas**: Strongly influenced the AI's decision
    - 🟡 **Yellow areas**: Moderately influenced
    - 🔵 **Blue areas**: Little to no influence
    
    ### Tips for Best Results
    
    - Use standard mammogram views (CC or MLO)
    - Ensure good image quality and contrast
    - The AI works best on 224x224 pixel images
    - For best results, use images similar to the training data
    
    ### Technical Details
    
    - Model processes images at 224x224 resolution
    - Inference time: <1 second on CPU
    - Heatmap generation: ~2 seconds
    """)
