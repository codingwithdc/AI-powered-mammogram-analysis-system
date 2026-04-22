# multiview.py
import torch
import numpy as np
from PIL import Image
import streamlit as st

class MultiViewAnalyzer:
    """
    Analyzes both CC and MLO views together
    """
    def __init__(self, model, device, transform):
        self.model = model
        self.device = device
        self.transform = transform
    
    def analyze_single_view(self, image):
        """Analyze one view"""
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            output = self.model(input_tensor)
            probs = torch.softmax(output, dim=1)
            pred_class = torch.argmax(probs).item()
            confidence = probs[0, pred_class].item()
        
        return {
            'pred_class': pred_class,
            'confidence': confidence,
            'class_name': 'MALIGNANT' if pred_class == 1 else 'BENIGN'
        }
    
    def analyze_both_views(self, cc_image, mlo_image):
        """Analyze both views and combine results"""
        
        # Analyze each view
        cc_result = self.analyze_single_view(cc_image)
        mlo_result = self.analyze_single_view(mlo_image)
        
        # Decision fusion strategies
        results = {
            'cc': cc_result,
            'mlo': mlo_result,
            'fusion_methods': {}
        }
        
        # Strategy 1: Average confidence
        avg_confidence = (cc_result['confidence'] + mlo_result['confidence']) / 2
        avg_pred = 1 if avg_confidence > 0.5 else 0
        results['fusion_methods']['average'] = {
            'pred_class': avg_pred,
            'confidence': avg_confidence,
            'class_name': 'MALIGNANT' if avg_pred == 1 else 'BENIGN'
        }
        
        # Strategy 2: Max confidence (take the most confident view)
        if cc_result['confidence'] > mlo_result['confidence']:
            results['fusion_methods']['max_confidence'] = cc_result
        else:
            results['fusion_methods']['max_confidence'] = mlo_result
        
        # Strategy 3: Consensus (both must agree)
        if cc_result['pred_class'] == mlo_result['pred_class']:
            results['fusion_methods']['consensus'] = {
                'pred_class': cc_result['pred_class'],
                'confidence': (cc_result['confidence'] + mlo_result['confidence']) / 2,
                'class_name': cc_result['class_name'],
                'agreement': True
            }
        else:
            results['fusion_methods']['consensus'] = {
                'pred_class': 0,  # Default to benign when disagreement
                'confidence': 0.5,
                'class_name': 'DISAGREEMENT - REVIEW NEEDED',
                'agreement': False
            }
        
        # Strategy 4: Weighted by confidence
        total_conf = cc_result['confidence'] + mlo_result['confidence']
        weight_cc = cc_result['confidence'] / total_conf
        weight_mlo = mlo_result['confidence'] / total_conf
        
        weighted_score = (cc_result['pred_class'] * weight_cc + 
                         mlo_result['pred_class'] * weight_mlo)
        weighted_pred = 1 if weighted_score > 0.5 else 0
        weighted_conf = max(cc_result['confidence'], mlo_result['confidence'])
        
        results['fusion_methods']['weighted'] = {
            'pred_class': weighted_pred,
            'confidence': weighted_conf,
            'class_name': 'MALIGNANT' if weighted_pred == 1 else 'BENIGN'
        }
        
        return results
    
    def generate_heatmap(self, image):
        """Generate Grad-CAM for a single view"""
        from pytorch_grad_cam import GradCAM
        from pytorch_grad_cam.utils.image import show_cam_on_image
        from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
        
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            output = self.model(input_tensor)
            pred_class = torch.argmax(output).item()
        
        target_layer = self.model.features[-4]
        cam = GradCAM(model=self.model, target_layers=[target_layer])
        targets = [ClassifierOutputTarget(pred_class)]
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0]
        
        # Create overlay
        img_np = np.array(image.resize((224, 224)))
        img_normalized = img_np.astype(np.float32) / 255.0
        overlay = show_cam_on_image(img_normalized, grayscale_cam, use_rgb=True)
        
        return overlay, grayscale_cam

def display_multiview_results(analyzer, cc_image, mlo_image):
    """Display results in a nice format"""
    
    results = analyzer.analyze_both_views(cc_image, mlo_image)
    
    # Generate heatmaps
    cc_heatmap, _ = analyzer.generate_heatmap(cc_image)
    mlo_heatmap, _ = analyzer.generate_heatmap(mlo_image)
    
    return results, cc_heatmap, mlo_heatmap