import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from pytorch_grad_cam import GradCAM, GradCAMPlusPlus
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from src.models.orchid_module import OrchidLitModule
import os
import argparse
import glob

def get_target_layers(model_name, model):
    """Get the target layers for Grad-CAM based on the model architecture.
    
    :param model_name: The name of the model architecture.
    :param model: The trained model.
    :return: A list of target layers for Grad-CAM.
    """
    if model_name in ["resnet50", "bilinear_cnn"]:
        # For ResNet50 and Bilinear CNN (which uses ResNet50 backbone)
        return [model.net.backbone.layer4[-1]]
    elif model_name == "inception_v3":
        # For InceptionV3
        return [model.net.backbone.Mixed_7c]
    elif model_name in ["vit_base_16", "mae_vit", "dino_vit", "ibot_vit"]:
        # For Vision Transformers
        return [model.net.backbone.blocks[-1].norm1]
    else:
        # Default to the last layer of the backbone
        return [list(model.net.backbone.children())[-1]]

def preprocess_image(image_path, image_size=224):
    """Preprocess an image for model input.
    
    :param image_path: The path to the image file.
    :param image_size: The target image size.
    :return: A tuple of (preprocessed_image, original_image_np).
    """
    # Load and resize image
    img = Image.open(image_path).convert('RGB')
    img = img.resize((image_size, image_size))
    
    # Convert to numpy array for visualization
    original_image_np = np.array(img) / 255.0
    
    # Preprocess for model input
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    input_tensor = preprocess(img).unsqueeze(0)  # Add batch dimension
    
    return input_tensor, original_image_np

def generate_gradcam(model, input_tensor, target_layers, target_class=None):
    """Generate Grad-CAM heatmap for the given input.
    
    :param model: The trained model.
    :param input_tensor: The preprocessed input tensor.
    :param target_layers: The target layers for Grad-CAM.
    :param target_class: The target class index. If None, uses the predicted class.
    :return: A tuple of (heatmap, predicted_class).
    """
    # Set model to evaluation mode
    model.eval()
    
    # Create Grad-CAM object
    cam = GradCAM(model=model.net, target_layers=target_layers)
    
    # If target_class is None, use the predicted class
    if target_class is None:
        with torch.no_grad():
            outputs = model(input_tensor)
            predicted_class = torch.argmax(outputs, dim=1).item()
    else:
        predicted_class = target_class
    
    # Set targets
    targets = [ClassifierOutputTarget(predicted_class)]
    
    # Generate CAM
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
    grayscale_cam = grayscale_cam[0, :]
    
    return grayscale_cam, predicted_class

def visualize_and_save_gradcam(image_path, model, model_name, save_dir, device):
    """Generate and save Grad-CAM visualization for a single image.
    
    :param image_path: The path to the input image.
    :param model: The trained model.
    :param model_name: The name of the model architecture.
    :param save_dir: The directory to save the visualization.
    :param device: The device to use for inference.
    """
    # Preprocess image
    input_tensor, original_image_np = preprocess_image(image_path)
    input_tensor = input_tensor.to(device)
    
    # Get target layers
    target_layers = get_target_layers(model_name, model)
    
    # Generate Grad-CAM
    grayscale_cam, predicted_class = generate_gradcam(model, input_tensor, target_layers)
    
    # Create visualization
    visualization = show_cam_on_image(original_image_np, grayscale_cam, use_rgb=True)
    
    # Save the visualization
    image_name = os.path.basename(image_path)
    save_path = os.path.join(save_dir, f"gradcam_{model_name}_{image_name}")
    
    # Convert to PIL Image and save
    Image.fromarray(visualization).save(save_path)
    
    print(f"Grad-CAM visualization saved to {save_path}")
    print(f"Predicted class: {predicted_class}")

def main(args):
    """Main function for Grad-CAM visualization.
    
    :param args: Command line arguments.
    """
    # Load the model from checkpoint
    model = OrchidLitModule.load_from_checkpoint(args.ckpt_path)
    
    # Get the device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Get model name from checkpoint
    model_name = model.hparams.net.model_name
    
    # Create save directory if it doesn't exist
    save_dir = "visualizations/gradcam"
    os.makedirs(save_dir, exist_ok=True)
    
    # Get image paths
    if os.path.isdir(args.image_path):
        # If input is a directory, get all image files
        image_paths = glob.glob(os.path.join(args.image_path, "*.jpg")) + glob.glob(os.path.join(args.image_path, "*.png")) + glob.glob(os.path.join(args.image_path, "*.jpeg"))
    else:
        # If input is a single file
        image_paths = [args.image_path]
    
    # Generate Grad-CAM for each image
    for image_path in image_paths[:args.num_images]:  # Limit to num_images
        visualize_and_save_gradcam(image_path, model, model_name, save_dir, device)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Grad-CAM visualization for orchid classification")
    parser.add_argument("--ckpt_path", type=str, required=True, help="Path to the model checkpoint")
    parser.add_argument("--image_path", type=str, required=True, help="Path to the input image or directory")
    parser.add_argument("--num_images", type=int, default=10, help="Number of images to process")
    args = parser.parse_args()
    main(args)