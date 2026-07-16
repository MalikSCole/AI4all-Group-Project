import os
import time
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score

class NutritionOrdinalMultiTaskDataset(Dataset):
    def __init__(self, dataframe, image_root, transform=None):
        self.df = dataframe.reset_index(drop=True)
        self.image_root = image_root
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        dish_id = self.df.loc[idx, "dish_id"]
        image_path = os.path.join(self.image_root, dish_id, "rgb.png")
        
        # Load and convert image
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            # Fallback to zero tensor if image fails to load
            image = Image.new("RGB", (224, 224), color=0)
            
        if self.transform:
            image = self.transform(image)

        class_label = int(self.df.loc[idx, "calorie_quantile_class"])

        # Ordinal encoding:
        # Low    = 0 -> [0.0, 0.0]
        # Medium = 1 -> [1.0, 0.0]
        # High   = 2 -> [1.0, 1.0]
        ordinal_label = torch.tensor([
            1.0 if class_label >= 1 else 0.0,
            1.0 if class_label >= 2 else 0.0
        ], dtype=torch.float32)

        regression_values = self.df.loc[
            idx,
            ["calories", "mass", "fat", "carb", "protein"]
        ].values.astype("float32")

        regression_values = torch.tensor(regression_values, dtype=torch.float32)

        return image, ordinal_label, regression_values, class_label

class SamResNet18MultiTask(nn.Module):
    def __init__(self, pretrained=True, freeze_backbone=True):
        super(SamResNet18MultiTask, self).__init__()
        # Load pre-trained ResNet18 backbone
        self.backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
        
        # Freeze backbone parameters if training on CPU / for fast feature extraction
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
                
        in_features = self.backbone.fc.in_features
        # Replace the backbone fc layer with identity to act as a feature extractor
        self.backbone.fc = nn.Identity()
        
        # Shared fully-connected layers
        self.shared_fc = nn.Linear(in_features, 128)
        self.dropout = nn.Dropout(0.3)
        
        # Outputs
        self.ordinal_head = nn.Linear(128, 2)
        self.regression_head = nn.Linear(128, 5)
        
    def forward(self, x):
        features = self.backbone(x)
        features = F.relu(self.shared_fc(features))
        features = self.dropout(features)
        
        ordinal_output = self.ordinal_head(features)
        regression_output = self.regression_head(features)
        
        return ordinal_output, regression_output

def ordinal_logits_to_class(ordinal_logits):
    probabilities = torch.sigmoid(ordinal_logits)
    # Class is determined by count of thresholds passed (sum of binary decisions)
    predictions = (probabilities >= 0.5).sum(dim=1)
    return predictions

def main():
    project_dir = os.path.dirname(os.path.dirname(__file__))
    dataset_dir = os.path.join(project_dir, 'AI4All Dataset')
    image_root = os.path.join(dataset_dir, 'imagery', 'realsense_overhead')
    models_dir = os.path.join(project_dir, 'models')
    os.makedirs(models_dir, exist_ok=True)
    
    # 1. Load data
    dish_nutrition_path = os.path.join(dataset_dir, 'dish_nutrition_values.csv')
    if not os.path.exists(dish_nutrition_path):
        print(f"[-] Error: Nutrition CSV file not found at: {dish_nutrition_path}")
        return
        
    print("[+] Loading and preprocessing data...")
    df_nutrition = pd.read_csv(dish_nutrition_path)
    
    # Filter to dishes with valid images in the local filesystem
    df_nutrition['image_exists'] = df_nutrition['dish_id'].apply(
        lambda x: os.path.exists(os.path.join(image_root, x, "rgb.png"))
    )
    df_valid = df_nutrition[df_nutrition['image_exists'] == True].copy()
    print(f" * Found {len(df_valid)} valid dishes with local RGB images.")
    
    if len(df_valid) == 0:
        print("[-] Error: No images found. Ensure the dataset images are located under AI4All Dataset/imagery/")
        return
        
    # Categorize into 3 quantiles
    df_valid["calorie_quantile_class"] = pd.qcut(
        df_valid["calories"],
        q=3,
        labels=[0, 1, 2]
    ).astype(int)
    
    # Scale regression targets
    regression_targets = ["calories", "mass", "fat", "carb", "protein"]
    scaler = StandardScaler()
    df_valid[regression_targets] = scaler.fit_transform(df_valid[regression_targets])
    
    # Train-test split
    train_df, test_df = train_test_split(df_valid, test_size=0.2, random_state=42, stratify=df_valid["calorie_quantile_class"])
    train_df, val_df = train_test_split(train_df, test_size=0.1, random_state=42, stratify=train_df["calorie_quantile_class"])
    
    print(f" * Split counts - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    
    # Datasets and loaders
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    train_dataset = NutritionOrdinalMultiTaskDataset(train_df, image_root, train_transform)
    val_dataset = NutritionOrdinalMultiTaskDataset(val_df, image_root, val_transform)
    test_dataset = NutritionOrdinalMultiTaskDataset(test_df, image_root, val_transform)
    
    # Using small batch size for CPU safety
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # 2. Model setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[+] Using device: {device}")
    
    # If using CPU, freeze the ResNet backbone for fast feature extraction
    freeze_backbone = (device.type == "cpu")
    if freeze_backbone:
        print("[+] CPU detected: freezing ResNet18 backbone to enable fast local training.")
    else:
        print("[+] GPU detected: training ResNet18 backbone end-to-end.")
        
    model = SamResNet18MultiTask(pretrained=True, freeze_backbone=freeze_backbone).to(device)
    
    # Loss functions & optimizer
    classification_criterion = nn.BCEWithLogitsLoss()
    regression_criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)
    
    # Training Loop
    epochs = 5  # Quick train locally, can be increased by the user
    print(f"[+] Training sam_model (ResNet18 Multi-Task) for {epochs} epochs...")
    
    best_val_acc = 0.0
    model_save_path = os.path.join(models_dir, 'sam_cnn_model.pth')
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        
        for images, ordinal_labels, regression_values, _ in train_loader:
            images = images.to(device)
            ordinal_labels = ordinal_labels.to(device)
            regression_values = regression_values.to(device)
            
            optimizer.zero_grad()
            
            ordinal_outputs, regression_outputs = model(images)
            
            # Loss computation
            loss_cls = classification_criterion(ordinal_outputs, ordinal_labels)
            loss_reg = regression_criterion(regression_outputs, regression_values)
            
            # Combine loss (weight = 0.5 for regression)
            loss = loss_cls + 0.5 * loss_reg
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * images.size(0)
            
        train_loss = train_loss / len(train_loader.dataset)
        
        # Validation
        model.eval()
        val_preds = []
        val_labels = []
        with torch.no_grad():
            for images, _, _, class_labels in val_loader:
                images = images.to(device)
                ordinal_outputs, _ = model(images)
                predicted_classes = ordinal_logits_to_class(ordinal_outputs)
                
                val_preds.extend(predicted_classes.cpu().numpy())
                val_labels.extend(class_labels.numpy())
                
        val_acc = accuracy_score(val_labels, val_preds)
        print(f" * Epoch {epoch+1:02d} | Train Loss: {train_loss:.4f} | Val Accuracy: {val_acc:.2%}")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            # Save checkpoints
            torch.save({
                'model_state_dict': model.state_dict(),
                'val_accuracy': val_acc,
                'scaler': scaler,
                'class_names': ['Low', 'Medium', 'High']
            }, model_save_path)
            
    print(f"\n[+] Training completed! Best validation accuracy: {best_val_acc:.2%}")
    print(f"[+] Saved best model checkpoints to: {model_save_path}")

if __name__ == '__main__':
    main()
