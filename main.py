import os
import torch
import numpy as np
from torch import nn
from datasets import Dataset, Image
from transformers import ViTForImageClassification, ViTImageProcessor
from transformers import Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import random
from tqdm import tqdm

# 設定隨機種子以確保可重複性
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

set_seed()

# 定義資料夾路徑
data_dir = "./人工智慧第四組資料夾"
ai_dir = os.path.join(data_dir, "ai")
real_dir = os.path.join(data_dir, "real")

# 計算每個類別的圖片數量
print("計算圖片數量...")
num_ai_images = len([f for f in os.listdir(ai_dir) if os.path.isfile(os.path.join(ai_dir, f))])
num_real_images = len([f for f in os.listdir(real_dir) if os.path.isfile(os.path.join(real_dir, f))])
total_images = num_ai_images + num_real_images

print(f"AI 圖片數量: {num_ai_images}")
print(f"真實圖片數量: {num_real_images}")
print(f"總圖片數量: {total_images}")

# 自動設定 FOCAL_LOSS_ALPHA 基於類別比例
# 使用反頻率作為權重，為類別不平衡提供補償
ai_weight = num_real_images / total_images
real_weight = num_ai_images / total_images
FOCAL_LOSS_ALPHA = {0: ai_weight, 1: real_weight}  # 0=AI, 1=Real

print(f"FOCAL_LOSS_ALPHA: {FOCAL_LOSS_ALPHA}")

# 實現 Focal Loss
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha  # 類別權重
        self.gamma = gamma  # 聚焦參數
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        ce_loss = nn.functional.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        
        # 應用 alpha 權重
        if self.alpha is not None:
            alpha_tensor = torch.tensor([self.alpha[0], self.alpha[1]], device=targets.device)
            at = alpha_tensor.gather(0, targets)
            focal_loss = at * (1 - pt) ** self.gamma * ce_loss
        else:
            focal_loss = (1 - pt) ** self.gamma * ce_loss
            
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

# 創建自定義 Trainer 類來使用 Focal Loss
class CustomTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.focal_loss = FocalLoss(
            alpha=FOCAL_LOSS_ALPHA,
            gamma=2.0
        )
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss = self.focal_loss(logits, labels)
        return (loss, outputs) if return_outputs else loss

# 預處理函數
def preprocess_data(example_batch):
    # 處理可能的格式不一致問題
    images = []
    for img in example_batch["image"]:
        # 確保所有圖像都是 RGB 格式
        if hasattr(img, 'mode') and img.mode == 'RGBA':
            img = img.convert('RGB')
        images.append(img)
    
    # 使用 ViT 圖像處理器處理圖像
    inputs = processor(
        images, 
        return_tensors="pt", 
        padding=True
    )
    
    # 標籤: ai=0, real=1
    inputs["labels"] = example_batch["label"]
    return inputs

# 評估指標計算函數
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='weighted')
    accuracy = accuracy_score(labels, predictions)
    
    # 計算混淆矩陣
    cm = confusion_matrix(labels, predictions)
    
    return {
        'accuracy': accuracy,
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'confusion_matrix': cm.tolist(),  # 將 NumPy 數組轉換為列表
    }

# 創建資料集
def create_dataset(val_ratio=0.2, test_ratio=0.1):
    print("建立資料集...")
    
    ai_images = []
    for f in tqdm(os.listdir(ai_dir), desc="載入 AI 圖片"):
        if os.path.isfile(os.path.join(ai_dir, f)):
            ai_images.append((os.path.join(ai_dir, f), 0))
    
    real_images = []
    for f in tqdm(os.listdir(real_dir), desc="載入真實圖片"):
        if os.path.isfile(os.path.join(real_dir, f)):
            real_images.append((os.path.join(real_dir, f), 1))
    
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
    
    print(f"訓練集: {len(train_images)} 張圖片")
    print(f"驗證集: {len(val_images)} 張圖片")
    print(f"測試集: {len(test_images)} 張圖片")
    
    def create_dataset_dict(image_list):
        return {
            "image": [img_path for img_path, _ in image_list],
            "label": [label for _, label in image_list]
        }
    
    train_dict = create_dataset_dict(train_images)
    val_dict = create_dataset_dict(val_images)
    test_dict = create_dataset_dict(test_images)
    
    train_dataset = Dataset.from_dict(train_dict).cast_column("image", Image())
    val_dataset = Dataset.from_dict(val_dict).cast_column("image", Image())
    test_dataset = Dataset.from_dict(test_dict).cast_column("image", Image())
    
    return train_dataset, val_dataset, test_dataset

# 視覺化混淆矩陣
def plot_confusion_matrix(cm, classes, title='confusion_matrix'):
    plt.figure(figsize=(8, 6))
    # 如果 cm 是列表，則將其轉換回 NumPy 數組
    if isinstance(cm, list):
        cm = np.array(cm)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.xlabel('predicted label')
    plt.ylabel('true label')
    plt.title(title)
    plt.savefig('./confusion_matrix.png')
    plt.close()

# 主函數
def main():
    print("開始 AI vs 真實圖片分類訓練...")
    
    # 載入預訓練模型和處理器
    global processor
    model_name = "google/vit-large-patch16-224-in21k"
    processor = ViTImageProcessor.from_pretrained(model_name)
    
    # 創建資料集
    train_dataset, val_dataset, test_dataset = create_dataset()

    # 預處理資料集
    print("預處理訓練資料集...")
    train_dataset = train_dataset.map(
        preprocess_data, 
        batched=True, 
        batch_size=8,
        remove_columns=["image"]
    )
    
    print("預處理驗證資料集...")
    val_dataset = val_dataset.map(
        preprocess_data, 
        batched=True, 
        batch_size=8,
        remove_columns=["image"]
    )
    
    print("預處理測試資料集...")
    test_dataset = test_dataset.map(
        preprocess_data, 
        batched=True, 
        batch_size=8,
        remove_columns=["image"]
    )

    # 設置資料格式以匹配模型輸入
    train_dataset.set_format("torch")
    val_dataset.set_format("torch")
    test_dataset.set_format("torch")

    # 載入模型
    print("載入 ViT 模型...")
    model = ViTForImageClassification.from_pretrained(
        model_name,
        num_labels=2,
        id2label={0: "AI", 1: "Real"},
        label2id={"AI": 0, "Real": 1}
    )

    # 訓練參數
    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=10,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        save_total_limit=3,  # 只保存最好的3個模型
        bf16=True,  # 啟用混合精度訓練
        #report_to="tensorboard",
        torch_compile=True
    )

    # 初始化 Trainer
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )

    # 開始訓練
    print("開始訓練...")
    trainer.train()

    # 保存最終模型
    trainer.save_model("./ai_vs_real_classifier")
    print("模型訓練完成並已儲存!")

    # 進行最終評估
    print("在測試集上進行評估...")
    test_results = trainer.evaluate(test_dataset)
    print(f"測試結果: {test_results}")
    
    # 視覺化混淆矩陣
    if 'eval_confusion_matrix' in test_results:
        cm = test_results['eval_confusion_matrix']
        plot_confusion_matrix(cm, classes=["AI", "Real"], title="Confusion Matrix")
        print("混淆矩陣已儲存至 './confusion_matrix.png'")

if __name__ == "__main__":
    main()