import os
import random
import sys
import logging

import yaml
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from torch import nn
from torch.nn import functional as F # Used in FocalLoss

from datasets import Dataset, Image as HFImage # Renamed to avoid clash with PIL.Image if used elsewhere
from transformers import ViTForImageClassification, ViTImageProcessor
from transformers import Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix


# Basic Logging Configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s'
)
logger = logging.getLogger(__name__)

# Load configuration
try:
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    logger.info("Configuration file 'config.yaml' loaded successfully.")
except FileNotFoundError:
    logger.error("Configuration file 'config.yaml' not found. Please ensure it exists in the current directory.")
    sys.exit(1)
except yaml.YAMLError as e:
    logger.error(f"Error parsing configuration file 'config.yaml': {e}")
    sys.exit(1)
except Exception as e:
    logger.error(f"An unexpected error occurred while loading 'config.yaml': {e}")
    sys.exit(1)


def set_seed(seed: int = 42):
    """
    Sets the random seeds for reproducibility across multiple libraries.

    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

set_seed()

# Define paths from config
data_dir = config["paths"]["data_dir"]
ai_dir = os.path.join(data_dir, "ai")
real_dir = os.path.join(data_dir, "real")

# 計算每個類別的圖片數量
logger.info("Calculating image counts...")
try:
    num_ai_images = len([f for f in os.listdir(ai_dir) if os.path.isfile(os.path.join(ai_dir, f))])
    num_real_images = len([f for f in os.listdir(real_dir) if os.path.isfile(os.path.join(real_dir, f))])
    total_images = num_ai_images + num_real_images

    logger.info(f"AI 圖片數量: {num_ai_images}")
    logger.info(f"真實圖片數量: {num_real_images}")
    logger.info(f"總圖片數量: {total_images}")

    # 自動設定 FOCAL_LOSS_ALPHA 基於類別比例
    # 使用反頻率作為權重，為類別不平衡提供補償
    if total_images > 0:
        ai_weight = num_real_images / total_images
        real_weight = num_ai_images / total_images
        FOCAL_LOSS_ALPHA = {0: ai_weight, 1: real_weight}  # 0=AI, 1=Real
        logger.info(f"FOCAL_LOSS_ALPHA: {FOCAL_LOSS_ALPHA}")
    else:
        logger.warning("Total images is zero, cannot calculate FOCAL_LOSS_ALPHA. Using default or unweighted.")
        FOCAL_LOSS_ALPHA = None # Or some default
except FileNotFoundError as e:
    logger.error(f"Error: One or more data directories not found: {e}. Please check paths in config.yaml.")
    sys.exit(1)
except Exception as e:
    logger.error(f"An error occurred during image count calculation: {e}")
    sys.exit(1)


# 實現 Focal Loss
class FocalLoss(nn.Module):
    """
    Implements Focal Loss for handling class imbalance.

    Focal Loss is designed to address class imbalance by down-weighting
    the loss assigned to well-classified examples.

    Attributes:
        alpha (dict or None): Weighting factor for each class. For example,
                              if for a binary classification problem, alpha={0:0.25, 1:0.75},
                              it assigns a weight of 0.25 to class 0 and 0.75 to class 1.
                              If None, no alpha weighting is applied.
        gamma (float): Focusing parameter. Higher values give more weight to
                       hard-to-classify examples.
        reduction (str): Specifies the reduction to apply to the output:
                         'none' | 'mean' | 'sum'.
                         'none': no reduction will be applied.
                         'mean': the sum of the output will be divided by the number of
                                 elements in the output.
                         'sum': the output will be summed.
    """
    def __init__(self, alpha: dict = None, gamma: float = 2.0, reduction: str = 'mean'):
        super(FocalLoss, self).__init__()
        if alpha is not None and not isinstance(alpha, dict):
            raise ValueError("alpha must be a dict (e.g., {0: weight_class_0, 1: weight_class_1}) or None")
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for Focal Loss.

        Args:
            inputs (torch.Tensor): Logits from the model (before Softmax).
                                   Shape: (batch_size, num_classes).
            targets (torch.Tensor): Ground truth labels.
                                    Shape: (batch_size).

        Returns:
            torch.Tensor: The calculated focal loss. Shape depends on `reduction`.
        """
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss) # Probability of the true class
        
        focal_term = (1 - pt) ** self.gamma
        
        if self.alpha is not None:
            # Ensure alpha is a tensor and correctly aligned with targets
            # Assumes alpha dict keys are class indices (0, 1, ...)
            alpha_weights = torch.tensor([self.alpha.get(i, 1.0) for i in range(inputs.size(1))], device=targets.device)
            at = alpha_weights.gather(0, targets)
            focal_loss = at * focal_term * ce_loss
        else:
            focal_loss = focal_term * ce_loss
            
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else: # 'none'
            return focal_loss

# 創建自定義 Trainer 類來使用 Focal Loss
class CustomTrainer(Trainer):
    """
    Custom Trainer that uses FocalLoss if FOCAL_LOSS_ALPHA is defined.
    The gamma for FocalLoss is hardcoded to 2.0 in this implementation.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if FOCAL_LOSS_ALPHA: # Use FocalLoss only if alpha is available
            logger.info(f"CustomTrainer Initialized with FocalLoss, alpha: {FOCAL_LOSS_ALPHA}, gamma: 2.0")
            self.focal_loss_fn = FocalLoss(
                alpha=FOCAL_LOSS_ALPHA,
                gamma=2.0 # Gamma is often set to 2.0
            )
        else:
            logger.info("CustomTrainer Initialized with default CrossEntropyLoss (FOCAL_LOSS_ALPHA not provided or invalid).")
            self.focal_loss_fn = None # Fallback to standard CE loss handled by Trainer

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        Computes the loss for training. Uses FocalLoss if initialized,
        otherwise defaults to the Trainer's standard loss computation.
        """
        if not self.focal_loss_fn:
            return super().compute_loss(model, inputs, return_outputs)

        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss = self.focal_loss_fn(logits, labels)
        return (loss, outputs) if return_outputs else loss

# 預處理函數
def preprocess_data(example_batch: dict) -> dict:
    """
    Preprocesses a batch of examples for the ViT model.

    Converts RGBA images to RGB and uses the ViTImageProcessor.

    Args:
        example_batch (dict): A dictionary containing 'image' (list of PIL Images)
                              and 'label' (list of labels).

    Returns:
        dict: A dictionary containing processed 'pixel_values' and 'labels'
              ready for the model.
    """
    images = []
    for img in example_batch["image"]:
        if hasattr(img, 'mode') and img.mode == 'RGBA':
            logger.debug(f"Converting image from RGBA to RGB.")
            img = img.convert('RGB')
        images.append(img)
    
    # Assumes 'processor' is a globally available ViTImageProcessor instance
    inputs = processor(
        images, 
        return_tensors="pt", 
        padding=True
    )
    
    inputs["labels"] = example_batch["label"]
    return inputs

# 評估指標計算函數
def compute_metrics(eval_pred: tuple) -> dict:
    """
    Computes evaluation metrics from model predictions.

    Args:
        eval_pred (tuple): A tuple containing predictions (logits) and true labels.

    Returns:
        dict: A dictionary of calculated metrics (accuracy, f1, precision, recall,
              confusion_matrix).
    """
    predictions_logits, labels = eval_pred
    predictions = np.argmax(predictions_logits, axis=1)
    
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average='weighted', zero_division=0
    )
    accuracy = accuracy_score(labels, predictions)
    cm = confusion_matrix(labels, predictions)
    
    return {
        'accuracy': accuracy,
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'confusion_matrix': cm.tolist(),
    }

# 創建資料集
def create_dataset(
    ai_dir_path: str, 
    real_dir_path: str, 
    val_ratio: float, 
    test_ratio: float
) -> tuple[Dataset | None, Dataset | None, Dataset | None]:
    """
    Creates train, validation, and test datasets from AI and Real image directories.

    Args:
        ai_dir_path (str): Path to the directory containing AI-generated images.
        real_dir_path (str): Path to the directory containing real images.
        val_ratio (float): Proportion of the data to use for validation.
        test_ratio (float): Proportion of the data to use for testing.

    Returns:
        tuple[Dataset | None, Dataset | None, Dataset | None]: 
            A tuple containing train, validation, and test datasets.
            Returns (None, None, None) on critical error (e.g., no images found).
    """
    logger.info("Creating dataset...")
    try:
        ai_images = []
        for f_name in tqdm(os.listdir(ai_dir_path), desc="Loading AI images"):
            if os.path.isfile(os.path.join(ai_dir_path, f_name)):
                ai_images.append((os.path.join(ai_dir_path, f_name), 0)) # label 0 for AI
        
        real_images = []
        for f_name in tqdm(os.listdir(real_dir_path), desc="Loading Real images"):
            if os.path.isfile(os.path.join(real_dir_path, f_name)):
                real_images.append((os.path.join(real_dir_path, f_name), 1)) # label 1 for Real
        
        if not ai_images and not real_images:
            logger.error("No images found in AI or Real directories. Cannot create dataset.")
            return None, None, None # Or raise an error

    except FileNotFoundError as e:
        logger.error(f"Error: Data directory not found during dataset creation: {e}. Please check paths.")
        return None, None, None # Or raise an error
    except Exception as e:
        logger.exception(f"An unexpected error occurred during image listing in create_dataset: {e}")
        return None, None, None

    # 打亂數據
    random.shuffle(ai_images)
    random.shuffle(real_images)
    
    # 計算分割索引
    ai_train_end = int(len(ai_images) * (1 - val_ratio - test_ratio))
    ai_val_end = int(len(ai_images) * (1 - test_ratio))
    
    real_train_end = int(len(real_images) * (1 - val_ratio - test_ratio))
    real_val_end = int(len(real_images) * (1 - test_ratio))
    
    # 分割資料集
    ai_train = ai_images[:ai_train_end]
    ai_val = ai_images[ai_train_end:ai_val_end]
    ai_test = ai_images[ai_val_end:]
    
    real_train = real_images[:real_train_end]
    real_val = real_images[real_train_end:real_val_end]
    real_test = real_images[real_val_end:]
    
    # 合併並打亂
    train_images = ai_train + real_train
    val_images = ai_val + real_val
    test_images = ai_test + real_test
    
    random.shuffle(train_images)
    random.shuffle(val_images)
    random.shuffle(test_images)

    logger.info(f"訓練集: {len(train_images)} 張圖片")
    logger.info(f"驗證集: {len(val_images)} 張圖片")
    logger.info(f"測試集: {len(test_images)} 張圖片")

    def create_dataset_dict(image_list):
        return {
            "image": [img_path for img_path, _ in image_list],
            "label": [label for _, label in image_list]
        }
    
    train_dict = create_dataset_dict(train_images)
    val_dict = create_dataset_dict(val_images)
    test_dict = create_dataset_dict(test_images)
    
    # Using HFImage (renamed import) to avoid potential conflicts
    train_dataset = Dataset.from_dict(train_dict).cast_column("image", HFImage())
    val_dataset = Dataset.from_dict(val_dict).cast_column("image", HFImage())
    test_dataset = Dataset.from_dict(test_dict).cast_column("image", HFImage())
    
    return train_dataset, val_dataset, test_dataset

# 視覺化混淆矩陣
def plot_confusion_matrix(cm_data: list | np.ndarray, class_names: list[str], save_path: str, title:str ='Confusion Matrix'):
    """
    Plots and saves a confusion matrix.

    Args:
        cm_data (list | np.ndarray): The confusion matrix data.
        class_names (list[str]): Names of the classes for labels.
        save_path (str): Path to save the confusion matrix image.
        title (str, optional): Title of the plot. Defaults to 'Confusion Matrix'.
    """
    plt.figure(figsize=(8, 6))
    if isinstance(cm_data, list):
        cm_data = np.array(cm_data)
    
    sns.heatmap(cm_data, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(title)
    
    try:
        plt.savefig(save_path)
        logger.info(f"Confusion matrix saved to '{save_path}'")
    except Exception as e:
        logger.exception(f"Error saving confusion matrix to '{save_path}': {e}")
    plt.close()

# 主函數
def main_training_pipeline():
    """
    Main function to run the AI vs Real image classification training pipeline.
    Orchestrates model loading, data creation, preprocessing, training, and evaluation.
    """
    logger.info("Starting AI vs Real image classification training script.")

    global processor # ViTImageProcessor instance, set after loading

    # Load configuration for paths and training parameters
    # Config is loaded at module level, accessible here.
    paths_config = config["paths"]
    training_params = config["training"]
    dataset_params = config["dataset"]

    try:
        model_name = paths_config["model_name"]
        logger.info(f"Loading ViTImageProcessor from '{model_name}'...")
        processor = ViTImageProcessor.from_pretrained(model_name)
        logger.info("ViTImageProcessor loaded successfully.")
    except Exception as e:
        logger.exception(f"Error loading ViTImageProcessor from '{model_name}': {e}")
        sys.exit(1)

    # Create datasets
    train_dataset, val_dataset, test_dataset = create_dataset(
        ai_dir_path=ai_dir, # ai_dir and real_dir are global, defined from config
        real_dir_path=real_dir,
        val_ratio=dataset_params["val_ratio"],
        test_ratio=dataset_params["test_ratio"]
    )
    if train_dataset is None or val_dataset is None or test_dataset is None: # Check if dataset creation failed
        logger.error("Dataset creation failed. Exiting.")
        sys.exit(1)


    # 預處理資料集
    preprocess_batch_size = config["dataset"]["preprocess_batch_size"]
    try:
        logger.info("Preprocessing training dataset...")
        train_dataset = train_dataset.map(
            preprocess_data,
            batched=True,
            batch_size=preprocess_batch_size,
            remove_columns=["image"]
        )
        logger.info("Training dataset preprocessed successfully.")

        logger.info("Preprocessing validation dataset...")
        val_dataset = val_dataset.map(
            preprocess_data,
            batched=True,
            batch_size=preprocess_batch_size,
            remove_columns=["image"]
        )
        logger.info("Validation dataset preprocessed successfully.")

        logger.info("Preprocessing test dataset...")
        test_dataset = test_dataset.map(
            preprocess_data,
            batched=True,
            batch_size=preprocess_batch_size,
            remove_columns=["image"]
        )
        logger.info("Test dataset preprocessed successfully.")
    except Exception as e:
        logger.exception(f"Error during dataset mapping/preprocessing: {e}")
        sys.exit(1)


    # 設置資料格式以匹配模型輸入
    try:
        train_dataset.set_format("torch")
        val_dataset.set_format("torch")
        test_dataset.set_format("torch")
        logger.info("Dataset formats set to 'torch'.")
    except Exception as e:
        logger.exception(f"Error setting dataset format: {e}")
        sys.exit(1)

    try:
        # 載入模型
        logger.info(f"Loading ViTForImageClassification model from {model_name_from_config}...")
        model = ViTForImageClassification.from_pretrained(
            model_name_from_config,
            num_labels=2,
            id2label={0: "AI", 1: "Real"},
            label2id={"AI": 0, "Real": 1}
        )
        logger.info("ViTForImageClassification model loaded successfully.")
    except Exception as e:
        logger.exception(f"Error loading ViTForImageClassification model from {model_name_from_config}: {e}")
        sys.exit(1)


    # 訓練參數
    # TrainingArguments uses training_params from config
    logger.info(f"Training arguments: {training_params}")
    training_args_obj = TrainingArguments(
        output_dir=paths_config["output_dir"],
        num_train_epochs=training_params["num_train_epochs"],
        per_device_train_batch_size=training_params["per_device_train_batch_size"],
        per_device_eval_batch_size=training_params["per_device_eval_batch_size"],
        warmup_steps=training_params["warmup_steps"],
        weight_decay=training_params["weight_decay"],
        logging_dir=paths_config["logging_dir"],
        logging_steps=training_params["logging_steps"],
        eval_strategy="epoch", # Evaluate at the end of each epoch
        save_strategy="epoch", # Save model at the end of each epoch
        load_best_model_at_end=True,
        metric_for_best_model="f1", # Use F1 score to determine the best model
        save_total_limit=training_params["save_total_limit"],
        bf16=training_params["bf16"],
        torch_compile=training_params["torch_compile"],
        report_to="none" # Disable default reporting (e.g. wandb, tensorboard) unless explicitly configured
    )

    try:
        # Initialize Trainer
        logger.info("Initializing CustomTrainer...")
        trainer = CustomTrainer(
            model=model, # model is loaded ViTForImageClassification
            args=training_args_obj,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=compute_metrics,
        )
        logger.info("CustomTrainer initialized successfully.")

        # Start training
        logger.info("Starting training...")
        trainer.train()
        logger.info("Training completed.")

    except Exception as e:
        logger.exception(f"An error occurred during trainer initialization or training: {e}")
        sys.exit(1)

    try:
        # Save final model
        final_model_save_path = paths_config["final_model_path"]
        logger.info(f"Saving final model to '{final_model_save_path}'...")
        trainer.save_model(final_model_save_path)
        logger.info(f"Model saved successfully to '{final_model_save_path}'")
    except Exception as e:
        logger.exception(f"Error saving the final model: {e}")
        # Continue to evaluation even if saving fails

    try:
        # Final evaluation on the test set
        logger.info("Evaluating on the test set...")
        test_results = trainer.evaluate(test_dataset)
        logger.info(f"Test results: {test_results}")
        
        if 'eval_confusion_matrix' in test_results:
            cm_data = test_results['eval_confusion_matrix']
            plot_confusion_matrix(
                cm_data, 
                classes=["AI", "Real"], 
                save_path=paths_config["confusion_matrix_path"],
                title="Test Set Confusion Matrix"
            )
        else:
            logger.warning("'eval_confusion_matrix' not found in test_results. Cannot plot confusion matrix.")
            
    except Exception as e:
        logger.exception(f"An error occurred during final evaluation or confusion matrix plotting: {e}")

    logger.info("Training script finished.")

if __name__ == "__main__":
    main_training_pipeline()