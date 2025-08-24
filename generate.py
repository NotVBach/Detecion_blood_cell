import json
import os
import shutil
from collections import Counter
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
import pandas as pd

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Paths
base_dir = 'noaug'
aug_dir = 'aug_gan'
train_json = os.path.join(base_dir, 'annotations', 'train.json')
val_json = os.path.join(base_dir, 'annotations', 'val.json')
test_json = os.path.join(base_dir, 'annotations', 'test.json')
train_img_dir = os.path.join(base_dir, 'train')
val_img_dir = os.path.join(base_dir, 'val')
test_img_dir = os.path.join(base_dir, 'test')

# Create aug directory structure, including logs for losses
os.makedirs(os.path.join(aug_dir, 'annotations'), exist_ok=True)
os.makedirs(os.path.join(aug_dir, 'train'), exist_ok=True)
os.makedirs(os.path.join(aug_dir, 'val'), exist_ok=True)
os.makedirs(os.path.join(aug_dir, 'test'), exist_ok=True)
os.makedirs(os.path.join(aug_dir, 'logs'), exist_ok=True)

# Copy original images and annotations to aug/
shutil.copytree(train_img_dir, os.path.join(aug_dir, 'train'), dirs_exist_ok=True)
shutil.copytree(val_img_dir, os.path.join(aug_dir, 'val'), dirs_exist_ok=True)
shutil.copytree(test_img_dir, os.path.join(aug_dir, 'test'), dirs_exist_ok=True)
shutil.copy(val_json, os.path.join(aug_dir, 'annotations', 'val.json'))
shutil.copy(test_json, os.path.join(aug_dir, 'annotations', 'test.json'))

# Load train.json
with open(train_json, 'r') as f:
    data = json.load(f)

images = data['images']
categories = data['categories']
annotations = data['annotations']

# Map category_id to category_name
cat_id_to_name = {cat['id']: cat['name'] for cat in categories}

# Analyze class distribution (assuming one annotation per image)
image_id_to_category = {ann['image_id']: ann['category_id'] for ann in annotations}
class_counts = Counter(image_id_to_category.values())
print("Original class distribution:", {cat_id_to_name[k]: v for k, v in class_counts.items()})

# Determine target count (maximum class count to balance)
target_count = max(class_counts.values())
to_generate = {k: target_count - v for k, v in class_counts.items() if v < target_count}
print("Images to generate per class:", {cat_id_to_name[k]: v for k, v in to_generate.items()})

# Custom Dataset for loading images by class
class DefectDataset(Dataset):
    def __init__(self, images, annotations, img_dir, transform=None):
        self.images = images
        self.annotations = annotations
        self.img_dir = img_dir
        self.transform = transform
        self.image_id_to_file = {img['id']: img['file_name'] for img in images}

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        ann = self.annotations[idx]
        img_id = ann['image_id']
        img_path = os.path.join(self.img_dir, self.image_id_to_file[img_id])
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        label = ann['category_id']
        return image, label

# Image transformations (updated for 128x128)
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# DCGAN Generator (updated for 128x128)
class Generator(nn.Module):
    def __init__(self, latent_dim=100, feature_maps=64, channels=3):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, feature_maps * 16, 4, 1, 0, bias=False),  # 4x4
            nn.BatchNorm2d(feature_maps * 16),
            nn.ReLU(True),
            nn.ConvTranspose2d(feature_maps * 16, feature_maps * 8, 4, 2, 1, bias=False),  # 8x8
            nn.BatchNorm2d(feature_maps * 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(feature_maps * 8, feature_maps * 4, 4, 2, 1, bias=False),  # 16x16
            nn.BatchNorm2d(feature_maps * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(feature_maps * 4, feature_maps * 2, 4, 2, 1, bias=False),  # 32x32
            nn.BatchNorm2d(feature_maps * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(feature_maps * 2, feature_maps, 4, 2, 1, bias=False),  # 64x64
            nn.BatchNorm2d(feature_maps),
            nn.ReLU(True),
            nn.ConvTranspose2d(feature_maps, channels, 4, 2, 1, bias=False),  # 128x128
            nn.Tanh()
        )

    def forward(self, x):
        return self.main(x)

# DCGAN Discriminator (updated for 128x128)
class Discriminator(nn.Module):
    def __init__(self, feature_maps=64, channels=3):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(channels, feature_maps, 4, 2, 1, bias=False),  # 64x64
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(feature_maps, feature_maps * 2, 4, 2, 1, bias=False),  # 32x32
            nn.BatchNorm2d(feature_maps * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(feature_maps * 2, feature_maps * 4, 4, 2, 1, bias=False),  # 16x16
            nn.BatchNorm2d(feature_maps * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(feature_maps * 4, feature_maps * 8, 4, 2, 1, bias=False),  # 8x8
            nn.BatchNorm2d(feature_maps * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(feature_maps * 8, feature_maps * 16, 4, 2, 1, bias=False),  # 4x4
            nn.BatchNorm2d(feature_maps * 16),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(feature_maps * 16, 1, 4, 1, 0, bias=False),  # 1x1
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.main(x).view(-1)

# Function to save losses to CSV
def save_losses_to_csv(g_losses, d_losses, class_name, save_dir):
    df = pd.DataFrame({
        'Epoch': range(1, len(g_losses) + 1),
        'Generator_Loss': g_losses,
        'Discriminator_Loss': d_losses
    })
    csv_path = os.path.join(save_dir, f'losses_class_{class_name}.csv')
    df.to_csv(csv_path, index=False)
    print(f"Saved losses to {csv_path}")

# Function to plot and save losses
def plot_and_save_losses(g_losses, d_losses, class_name, save_dir):
    plt.figure(figsize=(10, 5))
    plt.plot(g_losses, label='Generator Loss')
    plt.plot(d_losses, label='Discriminator Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Loss Curves for Class {class_name}')
    plt.legend()
    plt.grid(True)
    plot_path = os.path.join(save_dir, f'loss_plot_class_{class_name}.png')
    plt.savefig(plot_path)
    plt.close()
    print(f"Saved loss plot to {plot_path}")

# Training DCGAN with loss tracking
def train_dcgan(generator, discriminator, dataloader, class_name, num_epochs=200, latent_dim=100, device='cuda'):
    criterion = nn.BCELoss()
    g_optimizer = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    d_optimizer = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

    g_losses = []
    d_losses = []

    generator.train()
    discriminator.train()

    for epoch in range(num_epochs):
        epoch_g_loss = 0.0
        epoch_d_loss = 0.0
        num_batches = 0

        for real_imgs, _ in dataloader:
            batch_size = real_imgs.size(0)
            real_imgs = real_imgs.to(device)
            real_labels = torch.ones(batch_size).to(device)
            fake_labels = torch.zeros(batch_size).to(device)

            # Train Discriminator
            d_optimizer.zero_grad()
            real_preds = discriminator(real_imgs)
            d_real_loss = criterion(real_preds, real_labels)

            z = torch.randn(batch_size, latent_dim, 1, 1).to(device)
            fake_imgs = generator(z)
            fake_preds = discriminator(fake_imgs.detach())
            d_fake_loss = criterion(fake_preds, fake_labels)
            d_loss = d_real_loss + d_fake_loss
            d_loss.backward()
            d_optimizer.step()

            # Train Generator
            g_optimizer.zero_grad()
            fake_preds = discriminator(fake_imgs)
            g_loss = criterion(fake_preds, real_labels)
            g_loss.backward()
            g_optimizer.step()

            epoch_g_loss += g_loss.item()
            epoch_d_loss += d_loss.item()
            num_batches += 1

        # Average losses for the epoch
        epoch_g_loss /= num_batches
        epoch_d_loss /= num_batches
        g_losses.append(epoch_g_loss)
        d_losses.append(epoch_d_loss)

        print(f'Epoch [{epoch+1}/{num_epochs}] D Loss: {epoch_d_loss:.4f}, G Loss: {epoch_g_loss:.4f}')

    # Save losses to CSV
    save_losses_to_csv(g_losses, d_losses, class_name, os.path.join(aug_dir, 'logs'))
    # Plot and save losses
    plot_and_save_losses(g_losses, d_losses, class_name, os.path.join(aug_dir, 'logs'))

    return generator

# Generate images for a specific class
def generate_images(generator, num_images, latent_dim=100, device='cuda'):
    generator.eval()
    with torch.no_grad():
        z = torch.randn(num_images, latent_dim, 1, 1).to(device)
        fake_imgs = generator(z)
        fake_imgs = (fake_imgs * 0.5 + 0.5) * 255  # Denormalize to [0, 255]
        fake_imgs = fake_imgs.permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)
    return fake_imgs

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Train DCGAN and generate images for each underrepresented class
latent_dim = 100
new_images = []
new_annotations = annotations.copy()
max_image_id = max(img['id'] for img in images)
max_ann_id = max(ann['id'] for ann in annotations)

for cat_id, num_to_gen in to_generate.items():
    if num_to_gen <= 0:
        continue

    # Filter annotations for the current class
    class_annotations = [ann for ann in annotations if ann['category_id'] == cat_id]
    dataset = DefectDataset(images, class_annotations, train_img_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Initialize and train DCGAN
    class_name = cat_id_to_name[cat_id]
    generator = Generator(latent_dim=latent_dim).to(device)
    discriminator = Discriminator().to(device)
    generator = train_dcgan(generator, discriminator, dataloader, class_name, num_epochs=150, latent_dim=latent_dim, device=device)

    # Generate synthetic images
    fake_imgs = generate_images(generator, num_to_gen, latent_dim=latent_dim, device=device)

    # Save generated images and update annotations
    for i, img_array in enumerate(fake_imgs):
        max_image_id += 1
        max_ann_id += 1
        img_name = f'synthetic_{cat_id}_{i}.jpg'
        img_path = os.path.join(aug_dir, 'train', img_name)
        Image.fromarray(img_array).save(img_path)

        # Update images list (updated for 128x128)
        new_images.append({
            'file_name': img_name,
            'height': 128,
            'width': 128,
            'id': max_image_id
        })

        # Update annotations list (no bbox as per instruction)
        new_annotations.append({
            'id': max_ann_id,
            'image_id': max_image_id,
            'category_id': cat_id,
            'area': 0,
            'iscrowd': 0,
            'bbox': [],
            'segmentation': []
        })

# Update train.json
updated_data = {
    'images': images + new_images,
    'categories': categories,
    'annotations': new_annotations
}
with open(os.path.join(aug_dir, 'annotations', 'train.json'), 'w') as f:
    json.dump(updated_data, f, indent=4)

# Verify new class distribution
new_image_id_to_category = {ann['image_id']: ann['category_id'] for ann in new_annotations}
new_class_counts = Counter(new_image_id_to_category.values())
print("New class distribution:", {cat_id_to_name[k]: v for k, v in new_class_counts.items()})

# Optional: Visualize some generated images
def visualize_images(images, num_to_show=5):
    fig, axes = plt.subplots(1, num_to_show, figsize=(15, 3))
    for i in range(min(num_to_show, len(images))):
        axes[i].imshow(images[i])
        axes[i].axis('off')
    plt.show()

if new_images:
    visualize_images(fake_imgs[:5])