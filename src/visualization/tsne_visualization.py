import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from lightning import Trainer
from src.data.my_dataset_datamodule import MyDatasetDataModule
from src.models.orchid_module import OrchidLitModule
import hydra
from omegaconf import DictConfig
import os
import argparse

def extract_features(model, dataloader, device):
    """Extract features from the model for the given dataloader.
    
    :param model: The trained model.
    :param dataloader: The dataloader to extract features from.
    :param device: The device to use for inference.
    :return: A tuple of (features, labels).
    """
    model.eval()
    features_list = []
    labels_list = []
    
    with torch.no_grad():
        for batch in dataloader:
            x, y = batch
            x = x.to(device)
            features = model.net.get_features(x)
            features_list.append(features.cpu().numpy())
            labels_list.append(y.cpu().numpy())
    
    features = np.concatenate(features_list, axis=0)
    labels = np.concatenate(labels_list, axis=0)
    
    return features, labels

def visualize_tsne(features, labels, class_names, model_name, save_path):
    """Visualize features using t-SNE.
    
    :param features: The extracted features.
    :param labels: The corresponding labels.
    :param class_names: The names of the classes.
    :param model_name: The name of the model.
    :param save_path: The path to save the visualization.
    """
    # Standardize features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    # Apply t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
    features_tsne = tsne.fit_transform(features_scaled)
    
    # Plot t-SNE
    plt.figure(figsize=(12, 10))
    
    # 使用支持更多颜色的色图
    num_classes = len(class_names)
    if num_classes <= 10:
        cmap = 'tab10'
    elif num_classes <= 20:
        cmap = 'tab20'
    else:
        # 使用连续色图，适合更多类别
        cmap = 'rainbow'
    
    scatter = plt.scatter(features_tsne[:, 0], features_tsne[:, 1], c=labels, cmap=cmap, s=50, alpha=0.7)
    
    # Add legend
    unique_labels = np.unique(labels)
    if len(unique_labels) == len(class_names):
        # 创建自定义图例，确保每个类别都有图例项
        handles = []
        for i, class_name in enumerate(class_names):
            # 为每个类别创建一个散点标记作为图例项
            handle = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=scatter.cmap(scatter.norm(i)), 
                               markersize=10, label=class_name)
            handles.append(handle)
        plt.legend(handles, class_names, title="Orchid Varieties", bbox_to_anchor=(1.05, 1), loc='upper left')
    else:
        # 回退到自动生成图例
        plt.legend(*scatter.legend_elements(), title="Orchid Varieties", bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.title(f't-SNE Visualization of {model_name} Features', fontsize=16)
    plt.xlabel('t-SNE Component 1', fontsize=12)
    plt.ylabel('t-SNE Component 2', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"t-SNE visualization saved to {save_path}")

def main(args):
    """Main function for t-SNE visualization.
    
    :param args: Command line arguments.
    """
    # Load the checkpoint
    checkpoint = torch.load(args.ckpt_path)
    
    # Load the configuration from the checkpoint
    cfg = checkpoint['hyper_parameters']
    
    # Initialize the data module
    data_module = MyDatasetDataModule(
        data_dir="data/my_dataset/",
        batch_size=32,
        num_workers=0,
        pin_memory=True
    )
    
    # Setup the data module
    data_module.setup(stage='test')
    
    # Get the test dataloader
    test_dataloader = data_module.test_dataloader()
    
    # Get class names
    class_names = data_module.dataset.classes
    
    # Initialize the trainer
    trainer = Trainer(accelerator='auto')
    
    # Load the model from checkpoint
    model = OrchidLitModule.load_from_checkpoint(args.ckpt_path)
    
    # Get the device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Extract features
    features, labels = extract_features(model, test_dataloader, device)
    
    # Create save directory if it doesn't exist
    save_dir = "visualizations/tsne"
    os.makedirs(save_dir, exist_ok=True)
    
    # Get model name from checkpoint path or use default
    if 'model_name' in cfg.net:
        model_name = cfg.net.model_name
    else:
        model_name = "unknown_model"
    
    # Save path
    save_path = os.path.join(save_dir, f"tsne_{model_name}.png")
    
    # Visualize t-SNE
    visualize_tsne(features, labels, class_names, model_name, save_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="t-SNE visualization for orchid classification")
    parser.add_argument("--ckpt_path", type=str, required=True, help="Path to the model checkpoint")
    args = parser.parse_args()
    main(args)