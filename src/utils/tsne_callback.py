import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from lightning import Callback
import os
from typing import List, Optional

class TSNEVisualizationCallback(Callback):
    """A callback that generates t-SNE visualizations at regular intervals during training."""
    
    def __init__(self, every_n_epochs: int = 5, n_samples: int = 1000):
        """Initialize the TSNEVisualizationCallback.
        
        :param every_n_epochs: Generate visualization every N epochs.
        :param n_samples: Number of samples to use for visualization.
        """
        super().__init__()
        self.every_n_epochs = every_n_epochs
        self.n_samples = n_samples
        self.save_dir = "visualizations/tsne"
        os.makedirs(self.save_dir, exist_ok=True)
    
    def on_validation_epoch_end(self, trainer, pl_module):
        """Generate t-SNE visualization at the end of validation epoch."""
        if trainer.current_epoch % self.every_n_epochs == 0:
            self.generate_tsne(trainer, pl_module, trainer.current_epoch)
    
    def generate_tsne(self, trainer, pl_module, epoch):
        """Generate t-SNE visualization.
        
        :param trainer: The trainer instance.
        :param pl_module: The LightningModule instance.
        :param epoch: The current epoch number.
        """
        # Get the dataloader
        dataloader = trainer.datamodule.val_dataloader()
        
        # Extract features and labels
        features, labels = self.extract_features(pl_module, dataloader)
        
        # Get class names
        class_names = dataloader.dataset.dataset.classes
        
        # Get model name
        model_name = pl_module.hparams.net.model_name
        
        # Visualize t-SNE
        self.visualize_tsne(features, labels, class_names, model_name, epoch)
    
    def extract_features(self, model, dataloader):
        """Extract features from the model for the given dataloader.
        
        :param model: The trained model.
        :param dataloader: The dataloader to extract features from.
        :return: A tuple of (features, labels).
        """
        model.eval()
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
        
        # Subsample if needed
        if len(features) > self.n_samples:
            indices = np.random.choice(len(features), self.n_samples, replace=False)
            features = features[indices]
            labels = labels[indices]
        
        return features, labels
    
    def visualize_tsne(self, features, labels, class_names, model_name, epoch):
        """Visualize features using t-SNE.
        
        :param features: The extracted features.
        :param labels: The corresponding labels.
        :param class_names: The names of the classes.
        :param model_name: The name of the model.
        :param epoch: The current epoch number.
        """
        # Standardize features
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        
        # Apply t-SNE
        tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
        features_tsne = tsne.fit_transform(features_scaled)
        
        # Plot t-SNE
        plt.figure(figsize=(12, 10))
        scatter = plt.scatter(features_tsne[:, 0], features_tsne[:, 1], c=labels, cmap='tab10', s=50, alpha=0.7)
        
        # Add legend
        handles, _ = scatter.legend_elements()
        plt.legend(handles, class_names, title="Orchid Varieties", bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.title(f't-SNE Visualization of {model_name} Features (Epoch {epoch})', fontsize=16)
        plt.xlabel('t-SNE Component 1', fontsize=12)
        plt.ylabel('t-SNE Component 2', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save the plot
        save_path = os.path.join(self.save_dir, f"tsne_{model_name}_epoch_{epoch}.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"t-SNE visualization saved to {save_path}")
