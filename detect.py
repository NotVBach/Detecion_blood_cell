import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from torchvision.models import VGG16_Weights, ResNet50_Weights
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, accuracy_score
import numpy as np
import argparse

# Custom Dataset Class
class CustomImageDataset(Dataset):
    def __init__(self, json_file, image_dir, transform=None):
        with open(json_file) as f:
            self.data = json.load(f)
        self.image_dir = image_dir
        self.transform = transform
        self.images = self.data['images']
        self.annotations = self.data['annotations']
        # Map category_id to 0-6 (7 classes)
        self.category_map = {cat['id']: i for i, cat in enumerate(self.data['categories'][:7])}
        # Category names for confusion matrix and metrics
        self.category_names = [cat['name'] for cat in self.data['categories'][:7]]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_info = self.images[idx]
        img_path = os.path.join(self.image_dir, img_info['file_name'])
        image = Image.open(img_path).convert('RGB')
        
        # Get the first annotation for this image (assuming one label per image)
        image_id = img_info['id']
        annotation = next((ann for ann in self.annotations if ann['image_id'] == image_id), None)
        label = self.category_map[annotation['category_id']] if annotation else 0  # Default to 0 if no annotation

        if self.transform:
            image = self.transform(image)
        return image, label

# Plot and Save Loss
def plot_loss(train_losses, val_losses, output_dir, model_name):
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(train_losses) + 1), train_losses, label='Train Loss')
    plt.plot(range(1, len(val_losses) + 1), val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'{model_name} Loss Curve')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'loss_plot.png'))
    plt.close()

# Evaluate Model and Save Metrics
def evaluate_model(model, test_loader, device, output_dir, category_names):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Compute metrics for all 7 classes, even if some are missing
    precision, recall, f1, support = precision_recall_fscore_support(
        all_labels, all_preds, labels=range(len(category_names)), average=None, zero_division=0
    )
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='macro', zero_division=0
    )
    accuracy = accuracy_score(all_labels, all_preds)
    
    # Save metrics to CSV
    metrics_df = pd.DataFrame({
        'Class': category_names + ['Macro Average'],
        'Precision': list(precision) + [macro_precision],
        'Recall': list(recall) + [macro_recall],
        'F1-Score': list(f1) + [macro_f1],
        'Support': list(support) + [None]  # Support is not defined for macro average
    })
    metrics_df.to_csv(os.path.join(output_dir, 'metrics.csv'), index=False)
    
    # Save accuracy separately
    with open(os.path.join(output_dir, 'accuracy.txt'), 'w') as f:
        f.write(f"Accuracy: {accuracy:.4f}\n")
    
    # Compute and save confusion matrix
    cm = confusion_matrix(all_labels, all_preds, labels=range(len(category_names)))
    cm_df = pd.DataFrame(cm, index=category_names, columns=category_names)
    cm_df.to_csv(os.path.join(output_dir, 'confusion_matrix.csv'))
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=category_names, yticklabels=category_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
    plt.close()

# Training Function
def train_model(model, train_loader, val_loader, num_epochs, device, output_dir):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    train_losses, val_losses = [], []
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)
        
        epoch_train_loss = running_loss / len(train_loader.dataset)
        train_losses.append(epoch_train_loss)
        
        # Validation
        model.eval()
        running_loss = 0.0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                running_loss += loss.item() * images.size(0)
        
        epoch_val_loss = running_loss / len(val_loader.dataset)
        val_losses.append(epoch_val_loss)
        
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {epoch_train_loss:.4f}, Val Loss: {epoch_val_loss:.4f}")
    
    # Save loss to CSV
    loss_df = pd.DataFrame({'Epoch': range(1, num_epochs+1), 'Train_Loss': train_losses, 'Val_Loss': val_losses})
    loss_df.to_csv(os.path.join(output_dir, 'loss.csv'), index=False)
    
    # Plot and save loss
    plot_loss(train_losses, val_losses, output_dir, model.__class__.__name__)
    
    return model

# Main Function
def main(dataset_folder):
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Data transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Load datasets
    train_dataset = CustomImageDataset(
        json_file=os.path.join(dataset_folder, 'annotations', 'train.json'),
        image_dir=os.path.join(dataset_folder, 'train'),
        transform=transform
    )
    val_dataset = CustomImageDataset(
        json_file=os.path.join(dataset_folder, 'annotations', 'val.json'),
        image_dir=os.path.join(dataset_folder, 'val'),
        transform=transform
    )
    test_dataset = CustomImageDataset(
        json_file=os.path.join(dataset_folder, 'annotations', 'test.json'),
        image_dir=os.path.join(dataset_folder, 'test'),
        transform=transform
    )
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Initialize models
    models_dict = {
        'vgg16': models.vgg16(weights=VGG16_Weights.IMAGENET1K_V1),
        'resnet50': models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    }
    
    # Modify models for 7-class classification and evaluate
    for model_name, model in models_dict.items():
        if model_name == 'vgg16':
            model.classifier[6] = nn.Linear(4096, 7)
        elif model_name == 'resnet50':
            model.fc = nn.Linear(model.fc.in_features, 7)
        
        model = model.to(device)
        
        # Create output directory
        output_dir = os.path.join('output', os.path.basename(dataset_folder), model_name)
        os.makedirs(output_dir, exist_ok=True)
        
        # Train model
        print(f"Training {model_name}...")
        trained_model = train_model(model, train_loader, val_loader, num_epochs=1, device=device, output_dir=output_dir)
        
        # Save model
        torch.save(trained_model.state_dict(), os.path.join(output_dir, 'model.pth'))
        
        # Evaluate model
        print(f"Evaluating {model_name}...")
        evaluate_model(trained_model, test_loader, device, output_dir, test_dataset.category_names)
    
    print(f"Training and evaluation complete. Results saved in {os.path.join('output', os.path.basename(dataset_folder))}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train VGG and ResNet on custom dataset')
    parser.add_argument('--dataset_folder', type=str, default='noaug', help='Path to dataset folder')
    args = parser.parse_args()
    
    main(args.dataset_folder)