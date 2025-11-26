import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from src.data.my_dataset_datamodule import MyDatasetDataModule
from src.models.orchid_module import OrchidLitModule
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from PIL import Image
from torchvision import transforms
import glob

class MultiModelVisualizer:
    """A class to generate multi-model comparison visualizations."""
    
    def __init__(self):
        """Initialize the MultiModelVisualizer."""
        self.save_dir = "visualizations/multi_model"
        os.makedirs(self.save_dir, exist_ok=True)
        self.data_module = MyDatasetDataModule(
            data_dir="data/my_dataset/",
            batch_size=32,
            num_workers=4,
            pin_memory=True
        )
        self.data_module.setup(stage="test")
    
    def get_model_checkpoints(self):
        """Get all model checkpoints from the checkpoints directory.
        
        :return: A list of tuples (model_name, checkpoint_path).
        """
        checkpoints_dir = "outputs"
        model_checkpoints = []
        
        for root, dirs, files in os.walk(checkpoints_dir):
            for file in files:
                if file.endswith(".ckpt") and "best" in file:
                    model_name = None
                    for model_type in ["resnet50", "inception_v3", "vit_base_16", "bilinear_cnn", "mae_vit", "dino_vit", "ibot_vit"]:
                        if model_type in root:
                            model_name = model_type
                            break
                    if model_name:
                        checkpoint_path = os.path.join(root, file)
                        model_checkpoints.append((model_name, checkpoint_path))
        
        return model_checkpoints
    
    def load_model(self, checkpoint_path):
        """Load a model from checkpoint.
        
        :param checkpoint_path: Path to the model checkpoint.
        :return: The loaded model.
        """
        model = OrchidLitModule.load_from_checkpoint(checkpoint_path)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        model.eval()
        return model
    
    def extract_features(self, model, dataloader):
        """Extract features from the model for the given dataloader.
        
        :param model: The trained model.
        :param dataloader: The dataloader to extract features from.
        :return: A tuple of (features, labels).
        """
        features_list = []
        labels_list = []
        
        with torch.no_grad():
            for batch in dataloader:
                x, y = batch
                x = x.to(model.device)
                features = model.net.get_features(x)
                features_list.append(features.cpu().numpy())
                labels_list.append(y.cpu().numpy())
        
        features = np.concatenate(features_list, axis=0)
        labels = np.concatenate(labels_list, axis=0)
        
        return features, labels
    
    def generate_multi_model_tsne(self):
        """Generate t-SNE visualization for multiple models."""
        print("Generating multi-model t-SNE visualization...")
        
        # Get model checkpoints
        model_checkpoints = self.get_model_checkpoints()
        if not model_checkpoints:
            print("No model checkpoints found!")
            return
        
        # Get dataloader and class names
        dataloader = self.data_module.val_dataloader()
        class_names = dataloader.dataset.dataset.classes
        
        # Extract features for all models
        all_features = {}
        all_labels = {}
        
        for model_name, checkpoint_path in model_checkpoints:
            print(f"Extracting features for {model_name}...")
            model = self.load_model(checkpoint_path)
            features, labels = self.extract_features(model, dataloader)
            all_features[model_name] = features
            all_labels[model_name] = labels
        
        # Generate t-SNE for each model
        tsne_results = {}
        
        for model_name, features in all_features.items():
            print(f"Generating t-SNE for {model_name}...")
            scaler = StandardScaler()
            features_scaled = scaler.fit_transform(features)
            
            tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
            features_tsne = tsne.fit_transform(features_scaled)
            tsne_results[model_name] = features_tsne
        
        # Plot all t-SNE results in a grid
        num_models = len(tsne_results)
        rows = (num_models + 1) // 2
        cols = 2
        
        plt.figure(figsize=(15, 5 * rows))
        
        for i, (model_name, features_tsne) in enumerate(tsne_results.items()):
            plt.subplot(rows, cols, i + 1)
            labels = all_labels[model_name]
            scatter = plt.scatter(features_tsne[:, 0], features_tsne[:, 1], c=labels, cmap='tab10', s=50, alpha=0.7)
            plt.title(f'{model_name} Features', fontsize=14)
            plt.xlabel('t-SNE Component 1')
            plt.ylabel('t-SNE Component 2')
            plt.grid(True, alpha=0.3)
        
        # Add a common legend
        plt.figlegend(*scatter.legend_elements(), title="Orchid Varieties", loc="center right")
        plt.tight_layout()
        
        # Save the plot
        save_path = os.path.join(self.save_dir, "multi_model_tsne.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Multi-model t-SNE visualization saved to {save_path}")
    
    def get_target_layers(self, model_name, model):
        """Get the target layers for Grad-CAM based on the model architecture.
        
        :param model_name: The name of the model architecture.
        :param model: The trained model.
        :return: A list of target layers for Grad-CAM.
        """
        if model_name in ["resnet50", "bilinear_cnn"]:
            return [model.net.backbone.layer4[-1]]
        elif model_name == "inception_v3":
            return [model.net.backbone.Mixed_7c]
        elif model_name in ["vit_base_16", "mae_vit", "dino_vit", "ibot_vit"]:
            return [model.net.backbone.blocks[-1].norm1]
        else:
            return [list(model.net.backbone.children())[-1]]
    
    def preprocess_image(self, image_path, image_size=224):
        """Preprocess an image for model input.
        
        :param image_path: The path to the image file.
        :param image_size: The target image size.
        :return: A tuple of (preprocessed_image, original_image_np).
        """
        img = Image.open(image_path).convert('RGB')
        img = img.resize((image_size, image_size))
        original_image_np = np.array(img) / 255.0
        
        preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        input_tensor = preprocess(img).unsqueeze(0)
        
        return input_tensor, original_image_np
    
    def generate_multi_model_gradcam(self, image_path):
        """Generate Grad-CAM visualization for multiple models on the same image.
        
        :param image_path: Path to the input image.
        """
        print(f"Generating multi-model Grad-CAM visualization for image: {image_path}")
        
        # Get model checkpoints
        model_checkpoints = self.get_model_checkpoints()
        if not model_checkpoints:
            print("No model checkpoints found!")
            return
        
        # Preprocess image
        input_tensor, original_image_np = self.preprocess_image(image_path)
        
        # Generate Grad-CAM for each model
        gradcam_results = {}
        
        for model_name, checkpoint_path in model_checkpoints:
            print(f"Generating Grad-CAM for {model_name}...")
            model = self.load_model(checkpoint_path)
            device = model.device
            input_tensor = input_tensor.to(device)
            
            # Get target layers
            target_layers = self.get_target_layers(model_name, model)
            
            # Create Grad-CAM object
            cam = GradCAM(model=model.net, target_layers=target_layers)
            
            # Generate CAM
            targets = [ClassifierOutputTarget(0)]  # Target the first class
            grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
            grayscale_cam = grayscale_cam[0, :]
            
            # Create visualization
            visualization = show_cam_on_image(original_image_np, grayscale_cam, use_rgb=True)
            gradcam_results[model_name] = visualization
        
        # Plot all Grad-CAM results in a grid
        num_models = len(gradcam_results)
        rows = (num_models + 1) // 2
        cols = 2
        
        plt.figure(figsize=(15, 5 * rows))
        
        for i, (model_name, visualization) in enumerate(gradcam_results.items()):
            plt.subplot(rows, cols, i + 1)
            plt.imshow(visualization)
            plt.title(f'{model_name}', fontsize=14)
            plt.axis('off')
        
        plt.suptitle(f'Grad-CAM Comparison for Image: {os.path.basename(image_path)}', fontsize=16, y=0.95)
        plt.tight_layout()
        
        # Save the plot
        image_name = os.path.basename(image_path).split('.')[0]
        save_path = os.path.join(self.save_dir, f"multi_model_gradcam_{image_name}.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Multi-model Grad-CAM visualization saved to {save_path}")
    
    def generate_gradcam_comparison(self, num_images=5):
        """Generate Grad-CAM comparison for multiple images.
        
        :param num_images: Number of images to process.
        """
        # Get image paths
        image_dir = "data/my_dataset/"
        image_paths = []
        
        for root, dirs, files in os.walk(image_dir):
            for file in files:
                if file.endswith((".jpg", ".jpeg", ".png")):
                    image_paths.append(os.path.join(root, file))
                    if len(image_paths) >= num_images:
                        break
            if len(image_paths) >= num_images:
                break
        
        # Generate Grad-CAM comparison for each image
        for image_path in image_paths:
            self.generate_multi_model_gradcam(image_path)
    
    def generate_loss_curve_comparison(self):
        """Generate loss curve comparison for multiple models."""
        print("Generating loss curve comparison...")
        
        # This function would typically load loss curves from TensorBoard logs
        # For simplicity, we'll create a placeholder implementation
        # In a real scenario, you would use TensorBoard's API to extract loss curves
        
        # Placeholder data - replace with actual loss curves from logs
        models = ["resnet50", "inception_v3", "vit_base_16", "bilinear_cnn", "mae_vit", "dino_vit", "ibot_vit"]
        epochs = range(1, 21)
        
        # Generate synthetic loss curves for demonstration
        np.random.seed(42)
        loss_curves = {}
        
        for model in models:
            # Generate synthetic loss curve
            train_loss = 0.5 * np.exp(-0.1 * np.array(epochs)) + 0.1 * np.random.rand(len(epochs))
            val_loss = 0.6 * np.exp(-0.08 * np.array(epochs)) + 0.15 * np.random.rand(len(epochs))
            loss_curves[model] = (train_loss, val_loss)
        
        # Plot loss curves
        plt.figure(figsize=(12, 8))
        
        for model in models:
            train_loss, val_loss = loss_curves[model]
            plt.plot(epochs, train_loss, label=f'{model} Train', linestyle='--')
            plt.plot(epochs, val_loss, label=f'{model} Val')
        
        plt.title('Loss Curve Comparison Across Models', fontsize=16)
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Loss', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        
        # Save the plot
        save_path = os.path.join(self.save_dir, "multi_model_loss_curves.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Loss curve comparison saved to {save_path}")
    
    def generate_all_visualizations(self):
        """Generate all multi-model visualizations."""
        print("Generating all multi-model visualizations...")
        
        # Generate multi-model t-SNE
        self.generate_multi_model_tsne()
        
        # Generate multi-model Grad-CAM
        self.generate_gradcam_comparison(num_images=3)
        
        # Generate loss curve comparison
        self.generate_loss_curve_comparison()
        
        print("All multi-model visualizations generated!")

def main():
    """Main function to generate multi-model visualizations."""
    visualizer = MultiModelVisualizer()
    visualizer.generate_all_visualizations()

if __name__ == "__main__":
    main()
