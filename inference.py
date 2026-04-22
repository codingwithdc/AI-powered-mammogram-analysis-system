# inference.py
import torch
import argparse
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from models import get_model
from heatmap import generate_gradcam, overlay_heatmap
from fixed_data_loader import val_transform  # Reuse your existing transform
import os

def main(args):
    print("="*60)
    print("🔍 RUNNING INFERENCE WITH GRAD-CAM")
    print("="*60)
    
    # Load model
    print(f"\n📂 Loading model from: {args.weights}")
    if not os.path.exists(args.weights):
        print(f"❌ Model not found at {args.weights}")
        return
        
    model = get_model(args.weights, model_type=args.model_type)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    print(f"✅ Model loaded successfully on {device}")
    
    # Load and preprocess image
    print(f"\n📸 Loading image: {args.image}")
    if not os.path.exists(args.image):
        print(f"❌ Image not found at {args.image}")
        return
        
    image = Image.open(args.image).convert('RGB')
    
    # Use your existing transform
    input_tensor = val_transform(image).unsqueeze(0).to(device)
    
    # Run inference
    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.softmax(output, dim=1)
        pred_class = torch.argmax(probs).item()
        confidence = probs[0, pred_class].item()
    
    class_name = 'MALIGNANT' if pred_class == 1 else 'BENIGN'
    print(f"\n🎯 Prediction: {class_name}")
    print(f"📊 Confidence: {confidence:.2%}")
    
    # Generate heatmap
    print("\n🔥 Generating Grad-CAM heatmap...")
    grayscale_cam = generate_gradcam(
        model, 
        input_tensor, 
        model_type=args.model_type
    )
    
    # Overlay heatmap
    heatmap_vis = overlay_heatmap(
        image, 
        grayscale_cam, 
        transparency=args.transparency
    )
    
    # Display or save
    if args.save:
        # Save the heatmap
        Image.fromarray(heatmap_vis).save(args.save)
        print(f"💾 Heatmap saved to: {args.save}")
        
        # Also save individual components if requested
        if args.save_all:
            # Save original resized
            original_resized = image.resize((224, 224))
            original_resized.save(args.save.replace('.png', '_original.png'))
            
            # Save raw heatmap
            plt.imsave(args.save.replace('.png', '_raw.png'), grayscale_cam, cmap='jet')
            print(f"💾 Additional files saved with suffix _original.png and _raw.png")
    else:
        # Show results in a nice plot
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Original
        axes[0, 0].imshow(image.resize((224, 224)))
        axes[0, 0].set_title('Original Mammogram')
        axes[0, 0].axis('off')
        
        # Heatmap only
        axes[0, 1].imshow(grayscale_cam, cmap='jet')
        axes[0, 1].set_title('AI Attention Heatmap')
        axes[0, 1].axis('off')
        
        # Overlay
        axes[1, 0].imshow(heatmap_vis)
        axes[1, 0].set_title(f'Overlay - {class_name} ({confidence:.1%})')
        axes[1, 0].axis('off')
        
        # Info text
        axes[1, 1].axis('off')
        info_text = f"""
        ANALYSIS RESULTS
        ================
        
        Prediction: {class_name}
        Confidence: {confidence:.2%}
        
        Model: {args.model_type}
        Parameters: 98K
        
        Interpretation:
        {'⚠️ Suspicious patterns detected' if pred_class == 1 else '✅ Normal patterns'}
        
        The heatmap shows areas the AI focused on:
        🔴 Red = Strong influence
        🟡 Yellow = Moderate influence
        🔵 Blue = Little influence
        
        Note: This is an AI assistant.
        Always consult a radiologist.
        """
        axes[1, 1].text(0.1, 0.9, info_text, transform=axes[1, 1].transAxes,
                       fontsize=10, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        plt.show()
    
    print("\n✅ Inference complete!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run inference with Grad-CAM')
    parser.add_argument('--image', required=True, help='Path to input image')
    parser.add_argument('--weights', default='models/final_model_3103.pth', 
                       help='Path to model weights')
    parser.add_argument('--model_type', default='supertiny', 
                       choices=['supertiny', 'resnet50'],
                       help='Type of model to use')
    parser.add_argument('--save', help='Path to save heatmap (optional)')
    parser.add_argument('--save_all', action='store_true', 
                       help='Save all components (original, raw heatmap)')
    parser.add_argument('--transparency', type=float, default=0.5,
                       help='Heatmap transparency (0-1)')
    args = parser.parse_args()
    main(args)