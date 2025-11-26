import os
import torch
import hydra
import rootutils
import pandas as pd
import numpy as np
from lightning import LightningDataModule, LightningModule, Trainer
from omegaconf import DictConfig
import time
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.utils import RankedLogger
from src.data.my_dataset_datamodule import MyDatasetDataModule
from src.models.orchid_module import OrchidLitModule

log = RankedLogger(__name__, rank_zero_only=True)

class ModelEvaluator:
    """A class to evaluate multiple models and generate comparison results."""
    
    def __init__(self, config_path: str = "../configs", config_name: str = "eval.yaml"):
        """Initialize the ModelEvaluator.
        
        :param config_path: Path to the configuration files.
        :param config_name: Name of the configuration file.
        """
        self.config_path = config_path
        self.config_name = config_name
        self.evaluation_results = []
        self.save_dir = "evaluation_results"
        os.makedirs(self.save_dir, exist_ok=True)
    
    def get_model_checkpoints(self):
        """Get all model checkpoints from the checkpoints directory.
        
        :return: A list of tuples (model_name, checkpoint_path).
        """
        checkpoints_dir = "outputs"  # Default Hydra outputs directory
        model_checkpoints = []
        
        # Iterate through all subdirectories in outputs
        for root, dirs, files in os.walk(checkpoints_dir):
            for file in files:
                if file.endswith(".ckpt") and "best" in file:
                    # Extract model name from the directory structure
                    model_name = None
                    for model_type in ["resnet50", "inception_v3", "vit_base_16", "bilinear_cnn", "mae_vit", "dino_vit", "ibot_vit"]:
                        if model_type in root:
                            model_name = model_type
                            break
                    if model_name:
                        checkpoint_path = os.path.join(root, file)
                        model_checkpoints.append((model_name, checkpoint_path))
        
        return model_checkpoints
    
    def evaluate_model(self, model_name: str, checkpoint_path: str):
        """Evaluate a single model.
        
        :param model_name: Name of the model.
        :param checkpoint_path: Path to the model checkpoint.
        :return: A dictionary containing evaluation results.
        """
        log.info(f"Evaluating model: {model_name}")
        log.info(f"Checkpoint path: {checkpoint_path}")
        
        # Initialize data module
        data_module = MyDatasetDataModule(
            data_dir="data/my_dataset/",
            batch_size=32,
            num_workers=4,
            pin_memory=True
        )
        data_module.setup(stage="test")
        
        # Load the model
        model = OrchidLitModule.load_from_checkpoint(checkpoint_path)
        
        # Initialize trainer
        trainer = Trainer(
            accelerator="auto",
            devices=1,
            logger=False
        )
        
        # Evaluate the model
        start_time = time.time()
        trainer.test(model=model, datamodule=data_module)
        end_time = time.time()
        
        # Calculate inference speed
        test_dataloader = data_module.test_dataloader()
        num_samples = len(test_dataloader.dataset)
        inference_time = end_time - start_time
        samples_per_second = num_samples / inference_time
        
        # Get metrics
        metrics = trainer.callback_metrics
        
        # Generate confusion matrix
        self.generate_confusion_matrix(model, test_dataloader, model_name)
        
        # Create result dictionary
        result = {
            "model_name": model_name,
            "accuracy": metrics["test/acc"].item(),
            "precision": metrics["test/precision"].item(),
            "recall": metrics["test/recall"].item(),
            "f1_score": metrics["test/f1"].item(),
            "loss": metrics["test/loss"].item(),
            "samples_per_second": samples_per_second,
            "inference_time": inference_time,
            "num_samples": num_samples
        }
        
        log.info(f"Evaluation results for {model_name}: {result}")
        return result
    
    def generate_confusion_matrix(self, model, dataloader, model_name):
        """Generate and save confusion matrix for a model.
        
        :param model: The trained model.
        :param dataloader: The test dataloader.
        :param model_name: Name of the model.
        """
        model.eval()
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for batch in dataloader:
                x, y = batch
                x = x.to(model.device)
                outputs = model(x)
                preds = torch.argmax(outputs, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(y.cpu().numpy())
        
        # Generate confusion matrix
        cm = confusion_matrix(all_targets, all_preds)
        
        # Get class names
        class_names = dataloader.dataset.dataset.classes
        
        # Plot confusion matrix
        plt.figure(figsize=(12, 10))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
        disp.plot(cmap=plt.cm.Blues, xticks_rotation='vertical')
        plt.title(f'Confusion Matrix - {model_name}', fontsize=16)
        plt.tight_layout()
        
        # Save confusion matrix
        save_path = os.path.join(self.save_dir, f"confusion_matrix_{model_name}.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        log.info(f"Confusion matrix saved to {save_path}")
    
    def evaluate_all_models(self):
        """Evaluate all models and generate comparison results."""
        log.info("Starting evaluation of all models...")
        
        # Get all model checkpoints
        model_checkpoints = self.get_model_checkpoints()
        
        if not model_checkpoints:
            log.error("No model checkpoints found!")
            return
        
        # Evaluate each model
        for model_name, checkpoint_path in model_checkpoints:
            result = self.evaluate_model(model_name, checkpoint_path)
            self.evaluation_results.append(result)
        
        # Generate comparison results
        self.generate_comparison_results()
        
        log.info("Evaluation of all models completed!")
    
    def generate_comparison_results(self):
        """Generate comparison results and save to files."""
        # Create DataFrame from results
        df = pd.DataFrame(self.evaluation_results)
        
        # Sort by accuracy
        df = df.sort_values(by="accuracy", ascending=False)
        
        # Save to CSV
        csv_path = os.path.join(self.save_dir, "model_comparison.csv")
        df.to_csv(csv_path, index=False)
        log.info(f"Model comparison saved to {csv_path}")
        
        # Save to Markdown
        md_path = os.path.join(self.save_dir, "model_comparison.md")
        with open(md_path, "w") as f:
            f.write("# Model Comparison Results\n\n")
            f.write("## Quantitative Results\n\n")
            f.write(df.to_markdown(index=False, floatfmt=".4f"))
            f.write("\n\n")
        log.info(f"Model comparison (Markdown) saved to {md_path}")
        
        # Generate comparison plots
        self.generate_comparison_plots(df)
    
    def generate_comparison_plots(self, df: pd.DataFrame):
        """Generate comparison plots for the models.
        
        :param df: DataFrame containing evaluation results.
        """
        # Set model name as index for plotting
        df = df.set_index("model_name")
        
        # Metrics to plot
        metrics = ["accuracy", "precision", "recall", "f1_score"]
        
        # Plot accuracy, precision, recall, f1_score
        plt.figure(figsize=(12, 8))
        df[metrics].plot(kind="bar", figsize=(12, 8))
        plt.title("Model Performance Comparison", fontsize=16)
        plt.xlabel("Model", fontsize=12)
        plt.ylabel("Score", fontsize=12)
        plt.ylim(0, 1.0)
        plt.xticks(rotation=45)
        plt.legend(loc="upper right")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save the plot
        save_path = os.path.join(self.save_dir, "model_performance_comparison.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        log.info(f"Model performance comparison plot saved to {save_path}")
        
        # Plot inference speed
        plt.figure(figsize=(12, 6))
        df["samples_per_second"].plot(kind="bar", figsize=(12, 6), color="green")
        plt.title("Model Inference Speed Comparison", fontsize=16)
        plt.xlabel("Model", fontsize=12)
        plt.ylabel("Samples per Second", fontsize=12)
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save the plot
        save_path = os.path.join(self.save_dir, "model_inference_speed_comparison.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        log.info(f"Model inference speed comparison plot saved to {save_path}")

def main():
    """Main function to evaluate all models."""
    evaluator = ModelEvaluator()
    evaluator.evaluate_all_models()

if __name__ == "__main__":
    main()
