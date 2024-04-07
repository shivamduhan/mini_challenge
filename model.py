import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import models, transforms
from torchvision.models import vgg16, VGG16_Weights
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from PIL import Image
import cv2
from torch.cuda.amp import autocast, GradScaler
from facenet_pytorch import MTCNN
import torch.multiprocessing as mp
from torchvision.transforms.functional import to_pil_image
from torchvision.transforms.functional import to_pil_image, resize

scaler = GradScaler()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mtcnn = MTCNN(keep_all=False, select_largest=True, device=device)
def preprocess_dataset(csv_file, root_dir, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    labels_frame = pd.read_csv(csv_file)
    new_rows = []
    
    for idx, row in labels_frame.iterrows():
        img_name = os.path.join(root_dir, row[1])
        label = row[2]
        if not img_name.endswith('.jpg'):
            img_name += '.jpg'
        
        image = Image.open(img_name).convert('RGB')
        cropped_tensor = mtcnn(image)
        
        if cropped_tensor is not None:
            # Convert tensor to PIL Image before saving
            cropped_image = to_pil_image(cropped_tensor)
            save_path = os.path.join(save_dir, f"cropped_{idx}.jpg")
            cropped_image.save(save_path)
            new_rows.append([save_path, label])
        else:
            
            print(f"No face detected in {img_name}. Skipping...")
    
    new_df = pd.DataFrame(new_rows, columns=['image', 'label'])
    new_df.to_csv(os.path.join(save_dir, "preprocessed_labels.csv"), index=False)


preprocess_dataset('train_small.csv', 'train_small', 'preprocessed_train_tiny')


class PreprocessedCelebrityDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.labels_frame = pd.read_csv(csv_file)
        self.transform = transform

        
        unique_labels = sorted(self.labels_frame['label'].unique())
        self.label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
        self.idx_to_label = {idx: label for label, idx in self.label_to_idx.items()}

    def __len__(self):
        return len(self.labels_frame)

    def __getitem__(self, idx):
        img_name = self.labels_frame.iloc[idx, 0]
        label_str = self.labels_frame.iloc[idx, 1]
        label = self.label_to_idx[label_str]  # Convert label string to numeric

        image = Image.open(img_name).convert('RGB')
        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.long), img_name




transform = transforms.Compose([
    transforms.Resize((224, 224), antialias=True),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

    
transformed_dataset = PreprocessedCelebrityDataset(csv_file='preprocessed_train_tiny/preprocessed_labels.csv',
                                                   transform=transform)
dataset = transformed_dataset
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0, pin_memory=True)

vgg = vgg16(weights=VGG16_Weights.IMAGENET1K_V1)

for param in vgg.features.parameters():
    param.requires_grad = False

num_classes = len(dataset.label_to_idx)  


vgg.classifier[6] = nn.Linear(4096, num_classes)

# Check if CUDA is available and move the model to GPU
vgg.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(vgg.classifier.parameters(), lr=0.001)

num_epochs = 50
for epoch in range(num_epochs):
    vgg.train()
    total_loss = 0.0
    
    for inputs, labels, _ in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        
        with autocast():
            outputs = vgg(inputs)
            loss = criterion(outputs, labels)

        
        scaler.scale(loss).backward()

        
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}, Training Loss: {total_loss / len(train_loader)}")

    vgg.eval()
    val_total_loss = 0.0
    with torch.no_grad():
        for inputs, labels, _ in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)

        
            with autocast():
                outputs = vgg(inputs)
                val_loss = criterion(outputs, labels)

            val_total_loss += val_loss.item()

    print(f"Validation Loss: {val_total_loss / len(val_loader)}")


class TestCelebrityDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.mtcnn = MTCNN(keep_all=False, select_largest=True, device=self.device)
        self.img_names = [f for f in os.listdir(root_dir) if os.path.isfile(os.path.join(root_dir, f))]

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.img_names[idx])
        image = Image.open(img_name).convert('RGB')

        # Apply MTCNN for face detection
        cropped_tensor = self.mtcnn(image)

        # If a face is detected and cropped
        if cropped_tensor is not None:
            # Convert the tensor output from MTCNN back to PIL Image for consistent resizing
            image = to_pil_image(cropped_tensor)
        else:
            
            image = image

        # Resize the image to ensure uniform size
        image = resize(image, (224, 224)) 

        
        if not isinstance(image, torch.Tensor):
            image = transforms.ToTensor()(image)
        image = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(image)

        return image, img_name  

test_dataset = TestCelebrityDataset(
    root_dir='test', 
    transform=transforms.Compose([
        transforms.Resize((224, 224), antialias=True),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
)

test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0, pin_memory=True)


class_names = [label for idx, label in sorted(dataset.idx_to_label.items())]


vgg.eval()
image_ids = []
predicted_labels = []
with torch.no_grad():
    for inputs, paths in test_loader:
        inputs = inputs.to(device)
        outputs = vgg(inputs)
        _, predicted_indices = torch.max(outputs, 1)  # Get the indices of the max log-probability

        # Correctly extend the predicted_labels list with the celebrity names corresponding to the predicted indices
        predicted_labels.extend([class_names[idx] for idx in predicted_indices.cpu().numpy()])

        
        image_ids.extend([int(os.path.splitext(os.path.basename(path))[0]) for path in paths])


submission_df = pd.DataFrame({'Id': image_ids, 'Category': predicted_labels})
submission_df.sort_values(by='Id', inplace=True)
submission_df.reset_index(drop=True, inplace=True)

# Save the submission DataFrame to CSV
submission_df.to_csv('submission_try.csv', index=False)
