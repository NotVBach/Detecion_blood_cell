import collections
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from sklearn.neighbors import NearestNeighbors
import time
import os
import json
import shutil
from PIL import Image
import torchvision.transforms as transforms

print(torch.version.cuda)

t0 = time.time()

# Configuration for the autoencoder and Deep SMOTE
args = {
    'dim_h': 64,          # Size of hidden layers
    'n_channel': 3,       # Number of channels (3 for RGB, 1 for grayscale)
    'n_z': 300,           # Latent space dimensions
    'sigma': 1.0,         # Variance in latent space
    'lambda': 0.01,       # Discriminator loss weight
    'lr': 0.0002,         # Learning rate
    'epochs': 50,         # Number of training epochs
    'batch_size': 100,    # Batch size
    'save': True,         # Save model weights
    'train': True,        # Train the model
    'dataset': 'custom',  # Custom dataset
    'image_size': 64      # Target image size for resizing
}

# Define Encoder
class Encoder(nn.Module):
    def __init__(self, args):
        super(Encoder, self).__init__()
        self.n_channel = args['n_channel']
        self.dim_h = args['dim_h']
        self.n_z = args['n_z']
        
        self.conv = nn.Sequential(
            nn.Conv2d(self.n_channel, self.dim_h, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(self.dim_h, self.dim_h * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.dim_h * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(self.dim_h * 2, self.dim_h * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.dim_h * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(self.dim_h * 4, self.dim_h * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.dim_h * 8),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.fc = nn.Linear(self.dim_h * 8 * 4 * 4, self.n_z)  # For 64x64 input

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# Define Decoder (Fixed to output 64x64)
class Decoder(nn.Module):
    def __init__(self, args):
        super(Decoder, self).__init__()
        self.n_channel = args['n_channel']
        self.dim_h = args['dim_h']
        self.n_z = args['n_z']
        
        self.fc = nn.Sequential(
            nn.Linear(self.n_z, self.dim_h * 8 * 4 * 4),  # Start with 4x4 feature map
            nn.ReLU()
        )
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(self.dim_h * 8, self.dim_h * 4, 4, 2, 1, bias=False),  # 4x4 -> 8x8
            nn.BatchNorm2d(self.dim_h * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(self.dim_h * 4, self.dim_h * 2, 4, 2, 1, bias=False),  # 8x8 -> 16x16
            nn.BatchNorm2d(self.dim_h * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(self.dim_h * 2, self.dim_h, 4, 2, 1, bias=False),     # 16x16 -> 32x32
            nn.BatchNorm2d(self.dim_h),
            nn.ReLU(True),
            nn.ConvTranspose2d(self.dim_h, self.n_channel, 4, 2, 1, bias=False),     # 32x32 -> 64x64
            nn.Tanh()
        )

    def forward(self, x):
        x = self.fc(x)
        x = x.view(-1, self.dim_h * 8, 4, 4)  # Reshape to 4x4 feature map
        x = self.deconv(x)
        return x

# SMOTE functions
def biased_get_class(c, dec_x, dec_y):
    xbeg = dec_x[dec_y == c]
    ybeg = dec_y[dec_y == c]
    return xbeg, ybeg

def G_SM(X, y, n_to_sample, cl):
    n_neigh = 6
    nn = NearestNeighbors(n_neighbors=n_neigh, n_jobs=1)
    nn.fit(X)
    dist, ind = nn.kneighbors(X)
    
    base_indices = np.random.choice(list(range(len(X))), n_to_sample)
    neighbor_indices = np.random.choice(list(range(1, n_neigh)), n_to_sample)
    
    X_base = X[base_indices]
    X_neighbor = X[ind[base_indices, neighbor_indices]]
    
    samples = X_base + np.multiply(np.random.rand(n_to_sample, 1), X_neighbor - X_base)
    return samples, [cl] * n_to_sample

# Load COCO annotations
def load_coco_annotations(json_path):
    with open(json_path, 'r') as f:
        coco_data = json.load(f)
    return coco_data

# Create aug_smote directory structure
def create_aug_smote_structure(src_root, dst_root):
    os.makedirs(dst_root, exist_ok=True)
    for folder in ['train', 'val', 'test', 'annotations']:
        os.makedirs(os.path.join(dst_root, folder), exist_ok=True)
        if folder != 'annotations':
            src_folder = os.path.join(src_root, folder)
            dst_folder = os.path.join(dst_root, folder)
            for img in os.listdir(src_folder):
                shutil.copy(os.path.join(src_folder, img), os.path.join(dst_folder, img))
    for ann_file in ['train.json', 'val.json', 'test.json']:
        shutil.copy(os.path.join(src_root, 'annotations', ann_file),
                    os.path.join(dst_root, 'annotations', ann_file))

# Load and preprocess images
def load_images_and_labels(json_path, img_dir):
    coco_data = load_coco_annotations(json_path)
    images = coco_data['images']
    annotations = coco_data['annotations']
    categories = {cat['id']: cat['name'] for cat in coco_data['categories']}
    
    img_paths = []
    labels = []
    img_data = []
    
    transform = transforms.Compose([
        transforms.Resize((args['image_size'], args['image_size'])),
        transforms.ToTensor(),
        transforms.Normalize((0.5,) * args['n_channel'], (0.5,) * args['n_channel'])
    ])
    
    img_id_to_name = {img['id']: img['file_name'] for img in images}
    
    for ann in annotations:
        img_id = ann['image_id']
        cat_id = ann['category_id']
        if img_id in img_id_to_name:
            img_path = os.path.join(img_dir, img_id_to_name[img_id])
            img_paths.append(img_path)
            labels.append(cat_id)
    
    for img_path in img_paths:
        img = Image.open(img_path).convert('RGB' if args['n_channel'] == 3 else 'L')
        img = transform(img)
        img_data.append(img)
    
    img_data = torch.stack(img_data)
    labels = np.array(labels)
    return img_data, labels, img_id_to_name, categories

# Main execution
src_root = 'noaug'
dst_root = 'aug_smote'
create_aug_smote_structure(src_root, dst_root)

# Load training data
train_json = os.path.join(src_root, 'annotations', 'train.json')
train_img_dir = os.path.join(src_root, 'train')
dec_x, dec_y, img_id_to_name, categories = load_images_and_labels(train_json, train_img_dir)

print('Train images shape:', dec_x.shape)
print('Train labels shape:', dec_y.shape)
print('Class distribution:', collections.Counter(dec_y))

# Initialize models
device = 'cuda' if torch.cuda.is_available() else 'cpu'
encoder = Encoder(args).to(device)
decoder = Decoder(args).to(device)
criterion = nn.MSELoss().to(device)

# Train the autoencoder
if args['train']:
    enc_optim = torch.optim.Adam(encoder.parameters(), lr=args['lr'])
    dec_optim = torch.optim.Adam(decoder.parameters(), lr=args['lr'])
    train_loader = DataLoader(TensorDataset(dec_x, torch.tensor(dec_y, dtype=torch.long)),
                              batch_size=args['batch_size'], shuffle=True)
    
    best_loss = np.inf
    for epoch in range(args['epochs']):
        encoder.train()
        decoder.train()
        train_loss = 0.0
        
        for images, labs in train_loader:
            images, labs = images.to(device), labs.to(device)
            encoder.zero_grad()
            decoder.zero_grad()
            
            z_hat = encoder(images)
            x_hat = decoder(z_hat)
            mse = criterion(x_hat, images)
            
            tc = np.random.choice(list(categories.keys()), 1)[0]
            xclass, yclass = biased_get_class(tc, dec_x.numpy(), dec_y)
            if len(xclass) == 0:
                continue
            nsamp = min(len(xclass), 100)
            ind = np.random.choice(len(xclass), nsamp, replace=False)
            xclass = xclass[ind]
            xclass = torch.Tensor(xclass).to(device)
            xc_enc = encoder(xclass)
            xc_enc = xc_enc.detach()
            ximg = decoder(xc_enc)
            mse2 = criterion(ximg, xclass)
            
            comb_loss = mse + mse2
            comb_loss.backward()
            enc_optim.step()
            dec_optim.step()
            
            train_loss += comb_loss.item() * images.size(0)
        
        train_loss /= len(train_loader.dataset)
        print(f'Epoch: {epoch} \tTrain Loss: {train_loss:.6f}')
        
        if train_loss < best_loss and args['save']:
            torch.save(encoder.state_dict(), os.path.join(dst_root, 'bst_enc.pth'))
            torch.save(decoder.state_dict(), os.path.join(dst_root, 'bst_dec.pth'))
            best_loss = train_loss

# Load best models for generation
encoder.load_state_dict(torch.load(os.path.join(dst_root, 'bst_enc.pth')))
decoder.load_state_dict(torch.load(os.path.join(dst_root, 'bst_dec.pth')))
encoder.eval()
decoder.eval()

# Balance the dataset
class_counts = collections.Counter(dec_y)
max_count = max(class_counts.values())
print(f'Target count per class: {max_count}')

resx = []
resy = []
new_images = []
new_image_id = max([img['id'] for img in load_coco_annotations(train_json)['images']]) + 1

for cls in categories.keys():
    xclass, yclass = biased_get_class(cls, dec_x.numpy(), dec_y)
    if len(xclass) == 0:
        continue
    n_to_sample = max_count - len(xclass)
    if n_to_sample <= 0:
        continue
    print(f'Generating {n_to_sample} samples for class {categories[cls]}')
    
    xclass_t = torch.Tensor(xclass).to(device)
    xclass_enc = encoder(xclass_t).detach().cpu().numpy()
    xsamp, ysamp = G_SM(xclass_enc, yclass, n_to_sample, cls)
    
    xsamp_t = torch.Tensor(xsamp).to(device)
    ximg = decoder(xsamp_t).detach().cpu()
    
    transform_inv = transforms.Normalize((-1,) * args['n_channel'], (2,) * args['n_channel'])
    for i in range(ximg.shape[0]):
        img = transform_inv(ximg[i])
        img = transforms.ToPILImage()(img)
        img_name = f'synthetic_{new_image_id}.jpg'
        img.save(os.path.join(dst_root, 'train', img_name))
        new_images.append({
            'file_name': img_name,
            'height': args['image_size'],
            'width': args['image_size'],
            'id': new_image_id
        })
        resx.append(img)
        resy.append(cls)
        new_image_id += 1

# Update train.json with new images
train_json_data = load_coco_annotations(train_json)
train_json_data['images'].extend(new_images)
new_annotations = []
for i, cls in enumerate(resy):
    new_annotations.append({
        'id': len(train_json_data['annotations']) + i + 1,
        'image_id': train_json_data['images'][-(len(resy) - i)]['id'],
        'category_id': cls,
        'bbox': [],
        'area': 0,
        'iscrowd': 0,
        'segmentation': []
    })
train_json_data['annotations'].extend(new_annotations)

with open(os.path.join(dst_root, 'annotations', 'train.json'), 'w') as f:
    json.dump(train_json_data, f)

# Verify new class distribution
new_dec_y = np.array([ann['category_id'] for ann in train_json_data['annotations']])
print('New class distribution:', collections.Counter(new_dec_y))

t1 = time.time()
print(f'Total time (min): {(t1 - t0) / 60:.2f}')