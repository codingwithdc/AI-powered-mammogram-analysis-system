# heatmap.py
import torch
import cv2
import numpy as np
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from models import get_target_layer

def generate_gradcam(model, image_tensor, target_layer=None, model_type='supertiny'):
    """
    Generate Grad-CAM heatmap for a given image tensor using pytorch_grad_cam
    
    Args:
        model: PyTorch model (your SuperTinyMammoNet)
        image_tensor: Preprocessed image tensor (1, 3, 224, 224)
        target_layer: Specific layer to hook (auto-detected if None)
        model_type: 'supertiny' or 'resnet50'
    
    Returns:
        grayscale_cam: Raw CAM values (224, 224)
    """
    model.eval()
    
    # Auto-detect target layer if not provided
    if target_layer is None:
        target_layer = get_target_layer(model, model_type)
    
    # Initialize Grad-CAM (use_cuda parameter removed - it auto-detects)
    cam = GradCAM(
        model=model, 
        target_layers=[target_layer]
    )
    
    # Get the predicted class
    with torch.no_grad():
        outputs = model(image_tensor)
        predicted_class = outputs.argmax(dim=1).item()
    
    # Generate CAM for the predicted class
    targets = [ClassifierOutputTarget(predicted_class)]
    grayscale_cam = cam(input_tensor=image_tensor, targets=targets)[0]
    
    return grayscale_cam

def overlay_heatmap(original_image, grayscale_cam, transparency=0.5):
    """
    Overlay heatmap on original image.
    
    Args:
        original_image: PIL Image or numpy array (H, W, 3)
        grayscale_cam: Raw CAM values (H, W) from generate_gradcam
        transparency: Heatmap opacity (0-1)
    
    Returns:
        visualization: RGB image with overlay (224x224x3)
    """
    # Convert original to numpy and normalize to 0-1
    if isinstance(original_image, Image.Image):
        original_image = np.array(original_image)
    
    # Resize to 224x224 if needed
    if original_image.shape[:2] != (224, 224):
        original_image = cv2.resize(original_image, (224, 224))
    
    # Normalize to [0, 1]
    original_normalized = original_image.astype(np.float32) / 255.0
    
    # Create overlay using pytorch_grad_cam utility
    visualization = show_cam_on_image(
        original_normalized,
        grayscale_cam,
        use_rgb=True,
        image_weight=transparency
    )
    
    return visualization

def generate_and_save_heatmap(model, image_tensor, original_image, save_path, 
                              transparency=0.5, model_type='supertiny'):
    """
    One-liner to generate and save heatmap.
    """
    grayscale_cam = generate_gradcam(model, image_tensor, model_type=model_type)
    vis = overlay_heatmap(original_image, grayscale_cam, transparency)
    
    # Save (convert RGB to BGR for OpenCV)
    cv2.imwrite(save_path, cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
    return vis

def generate_multiple_heatmaps(model, image_tensors, original_images, 
                               model_type='supertiny', transparency=0.5):
    """
    Generate heatmaps for multiple images
    """
    results = []
    target_layer = get_target_layer(model, model_type)
    
    cam = GradCAM(
        model=model,
        target_layers=[target_layer]
    )
    
    # Get predictions for all images
    with torch.no_grad():
        outputs = model(image_tensors)
        predicted_classes = outputs.argmax(dim=1)
    
    # Generate CAMs for all images
    targets = [ClassifierOutputTarget(cls) for cls in predicted_classes]
    grayscale_cams = cam(input_tensor=image_tensors, targets=targets)
    
    for i, (img, cam_val) in enumerate(zip(original_images, grayscale_cams)):
        vis = overlay_heatmap(img, cam_val, transparency)
        results.append(vis)
    
    return results