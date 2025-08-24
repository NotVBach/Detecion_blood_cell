import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from tqdm import tqdm
import argparse
import numpy as np

# Custom Dataset Class
class DefectDataset(Dataset):
    def __init__(self, root_dir, annotation_file, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        with open(annotation_file) as f:
            self.data = json.load(f)
        self.images = {img['id']: img for img in self.data['images']}
        self.annotations = self.data['annotations']
        # Map image_id to the first annotation's category_id (assuming one label per image)
        self.image_to_label = {}
        for ann in self.annotations:
            image_id = ann['image_id']
            if image_id not in self.image_to_label:  # Take first annotation
                self.image_to_label[image_id] = ann['category_id'] - 1  # Adjust to 0-based indexing
        self.image_ids = list(self.image_to_label.keys())

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        img_info = self.images[image_id]
        img_path = os.path.join(self.root_dir, img_info['file_name'])
        image = Image.open(img_path).convert('RGB')
        label = self.image_to_label[image_id]

        if self.transform:
            image = self.transform(image)
        return image, label

# Function to initialize model
def initialize_model(model_name, num_classes=7):
    if model_name.lower() == 'vgg':
        model = models.vgg16(weights='IMAGENET1K_V1')
        model.classifier[6] = nn.Linear(4096, num_classes)
    elif model_name.lower() == 'resnet':
        model = models.resnet50(weights='IMAGENET1K_V2')
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    else:
        raise ValueError("Model must be 'vgg' or 'resnet'")
    return model

# Training function
def train_model(model, train_loader, val_loader, device, model_name, output_dir, epochs=50):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    train_losses, val_losses = [], []
    best_val_loss = float('inf')

    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
            val_loss /= len(val_loader)
            val_losses.append(val_loss)

        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        scheduler.step()

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(output_dir, f'{model_name}_best.pth'))

    # Plot and save loss
    plt.figure()
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'{model_name} Loss Curve')
    plt.legend()
    plt.savefig(os.path.join(output_dir, f'{model_name}_loss_plot.png'))
    plt.close()

    return train_losses, val_losses

# Evaluation function
def evaluate_model(model, test_loader, device, model_name, output_dir, class_names):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted')

    # Save metrics
    with open(os.path.join(output_dir, f'{model_name}_metrics.txt'), 'w') as f:
        f.write(f"Accuracy: {accuracy:.4f}\n")
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall: {recall:.4f}\n")
        f.write(f"F1 Score: {f1:.4f}\n")

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'{model_name} Confusion Matrix')
    plt.savefig(os.path.join(output_dir, f'{model_name}_confusion_matrix.png'))
    plt.close()

    return accuracy, precision, recall, f1

# Main function
def main():
    parser = argparse.ArgumentParser(description="Train VGG or ResNet on defect dataset")
    parser.add_argument('--dataset_folder', type=str, default='noaug', help='Path to dataset folder')
    args = parser.parse_args()

    dataset_folder = args.dataset_folder
    output_base = 'output'
    os.makedirs(output_base, exist_ok=True)
    output_dir = os.path.join(output_base, os.path.basename(dataset_folder))
    os.makedirs(output_dir, exist_ok=True)

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Data transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load datasets
    train_dataset = DefectDataset(
        os.path.join(dataset_folder, 'train'),
        os.path.join(dataset_folder, 'annotations', 'train.json'),
        transform=transform
    )
    val_dataset = DefectDataset(
        os.path.join(dataset_folder, 'val'),
        os.path.join(dataset_folder, 'annotations', 'val.json'),
        transform=transform
    )
    test_dataset = DefectDataset(
        os.path.join(dataset_folder, 'test'),
        os.path.join(dataset_folder, 'annotations', 'test.json'),
        transform=transform
    )

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)
    test_loader = DataLoader(test_dataset, batch_size=32)

    # Class names (7 classes, ignoring 'platelet')
    class_names = ['ba', 'eo', 'erb', 'ig', 'lym', 'mono', 'neut']

    # Train and evaluate VGG
    vgg_output_dir = os.path.join(output_dir, 'vgg')
    os.makedirs(vgg_output_dir, exist_ok=True)
    vgg_model = initialize_model('vgg', num_classes=7).to(device)
    train_losses, val_losses = train_model(vgg_model, train_loader, val_loader, device, 'vgg', vgg_output_dir)
    accuracy, precision, recall, f1 = evaluate_model(vgg_model, test_loader, device, 'vgg', vgg_output_dir, class_names)
    print(f"VGG - Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")

    # Train and evaluate ResNet
    resnet_output_dir = os.path.join(output_dir, 'resnet')
    os.makedirs(resnet_output_dir, exist_ok=True)
    resnet_model = initialize_model('resnet', num_classes=7).to(device)
    train_losses, val_losses = train_model(resnet_model, train_loader, val_loader, device, 'resnet', resnet_output_dir)
    accuracy, precision, recall, f1 = evaluate_model(resnet_model, test_loader, device, 'resnet', resnet_output_dir, class_names)
    print(f"ResNet - Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")

if __name__ == '__main__':
    main()