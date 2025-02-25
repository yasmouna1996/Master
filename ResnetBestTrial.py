import torch
import monai
from monai.transforms import Compose, Resize
from monai.networks.nets import ResNet
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import torchio as tio
import optuna
from training_logger import get_logger
from optuna.pruners import MedianPruner
import os

trainingLogger = get_logger("trainingLogger")

# Dataset Class
class NiftiDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths  # List of paths to NIfTI files
        self.labels = labels  # Corresponding labels (e.g., binary classification labels)
        self.transform = transform  # Transformations (e.g., resizing, normalization)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        img = tio.ScalarImage(image_path)

        crop_or_pad = tio.CropOrPad((512,512,512))
        img = crop_or_pad(img)

        #apply resampling 256x256x256
        resample = tio.Resample((2,2,2))
        img = resample(img)


        if self.transform:
            img = self.transform(img)

        # Convert to tensor and ensure contiguity
        img_data = torch.tensor(img.data, dtype=torch.float32).contiguous()
        return img_data, torch.tensor(label, dtype=torch.long)

# Load the Excel file
excel_file = r"W:\\strahlenklinik\\science\\AI_Radiation\\Yasmine\\Dataset_Labels.xlsx"
df = pd.read_excel(excel_file)

# Path to your NIfTI files
dataset_dir = Path(r"W:\\strahlenklinik\\science\\AI_Radiation\\Yasmine\\Distortion-Correction-Dataset\\ReOriented\\UnBiasField")

# Get all NIfTI image paths
image_paths = list(dataset_dir.glob("*.nii*"))

# Load the labels from the Excel file
labels_df = pd.read_excel(excel_file)
labels_df.columns = ['PatientName', 'Class', 'Gradient Coil']
labels_df['Class'] = labels_df['Class'].replace({'ND': 1, '2D': 0})

# Filter out images that don't have a label
valid_images = [(image_path, label) for image_path, label in zip(image_paths, labels_df['Class'])]
valid_image_paths, valid_labels = zip(*valid_images)

# Shuffle and split the data into training, validation, and test sets
train_image_paths, temp_image_paths, train_labels, temp_labels = train_test_split(
    valid_image_paths, valid_labels, test_size=0.3, random_state=42
)

val_image_paths, test_image_paths, val_labels, test_labels = train_test_split(
    temp_image_paths, temp_labels, test_size=0.5, random_state=42
)

# Ensure variables are lists
train_image_paths = list(train_image_paths)
val_image_paths = list(val_image_paths)
test_image_paths = list(test_image_paths)
train_labels = list(train_labels)
val_labels = list(val_labels)
test_labels = list(test_labels)

# Set device (GPU if available, otherwise CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Hyperparameter suggestions
lr = 4.763308249591026e-05
batch_size = 1
dropout_rate = 0.1788625302214042
#noise_prob = 0.4336880
flip_prob = 0.4362906859550727

trainingLogger.info(f'Learning Rate: {lr}')
trainingLogger.info(f'batch size: {batch_size}')
trainingLogger.info(f'dropout rate: {dropout_rate}')
#trainingLogger.info(f'noise prob:{noise_prob}')
trainingLogger.info(f'flip prob:{flip_prob}')

# Data augmentation transforms
train_transform = tio.Compose([
    tio.RescaleIntensity(percentiles=(0.5, 99.5)),
    tio.ZNormalization(),
    #tio.Resample((1, 1, 1.5)),
    #tio.CropOrPad((160, 240, 160)),
    #tio.RandomNoise(std=0.2, p=noise_prob),
    tio.RandomFlip(flip_probability=flip_prob)
])

val_transform = tio.Compose([
    tio.RescaleIntensity(percentiles=(0.5, 99.5)),
    tio.ZNormalization(),
    #tio.Resample((1, 1, 1.5)),
    #tio.CropOrPad((160, 240, 160))
])

# Datasets and DataLoaders
train_dataset = NiftiDataset(train_image_paths, train_labels, transform=train_transform)
val_dataset = NiftiDataset(val_image_paths, val_labels, transform=val_transform)
test_dataset = NiftiDataset(test_image_paths, test_labels, transform=val_transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Model
model = ResNet(
    spatial_dims=3,
    n_input_channels=1,
    num_classes=2,
    block='basic',
    layers=[2, 2, 2, 2],
    block_inplanes=[64, 128, 256, 512],
).to(device)

save_dir = "./saved_models_bestTrial"
os.makedirs(save_dir, exist_ok=True)  # Create the directory if it doesn't exist

#Loss and optimizer
loss_function = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# Initialize variables to store the best models
best_epochs = []  # List to store (accuracy, epoch number, model state dict)

# Training loop
for epoch in range(50):  # Fewer epochs for faster trials
    model.train()
    running_loss = 0.0  # Reset loss per epoch
    correct_predictions = 0  # Reset correct predictions per epoch
    total_predictions = 0  # Reset total predictions per epoch

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        images = images.contiguous()  # Ensure the tensor is contiguous

        optimizer.zero_grad()
        outputs = model(images)

        # Compute loss
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()  # Track loss

        # Compute accuracy
        _, predicted = torch.max(outputs, 1)  # Get class with highest probability
        correct_predictions += (predicted == labels).sum().item()  # Count correct predictions
        total_predictions += labels.size(0)  # Total samples

    # Compute epoch metrics
    epoch_loss = running_loss / len(train_loader)
    accuracy = correct_predictions / total_predictions  # Avoid division by zero
    trainingLogger.info(f"Epoch {epoch+1}/50, Loss: {epoch_loss:.4f}, Accuracy: {accuracy:.4f}")
    print(f"Epoch {epoch+1}/50, Loss: {epoch_loss:.4f}, Accuracy: {accuracy:.4f}")
    #########################################################################################################################################
   # Track the top 3 best models based on validation accuracy
    model.eval()  # Switch to evaluation mode
    val_correct_predictions = 0
    with torch.no_grad():
        for val_images, val_labels in val_loader:
            val_images, val_labels = val_images.to(device), val_labels.to(device)
            val_images = val_images.contiguous()  # Ensure the tensor is contiguous
            outputs = model(val_images)
            _, predicted = torch.max(outputs, 1)
            val_correct_predictions += (predicted == val_labels).sum().item()

        val_accuracy = val_correct_predictions / len(val_loader.dataset)
        trainingLogger.info(f"Validation Accuracy: {val_accuracy:.4f}")

        # Save the model if it's one of the top 3
        if len(best_epochs) < 3:
            best_epochs.append((val_accuracy, epoch, model.state_dict()))
        else:
            # Check if this epoch has a better validation accuracy
            best_epochs.sort(reverse=True, key=lambda x: x[0])  # Sort by accuracy, descending
            if val_accuracy > best_epochs[-1][0]:  # If this epoch has better accuracy than the worst of the top 3
                best_epochs[-1] = (val_accuracy, epoch, model.state_dict())  # Replace the worst model

# After training, save the top 3 best models
for idx, (val_accuracy, epoch, model_state_dict) in enumerate(best_epochs):
    save_path = os.path.join(save_dir, f"best_epoch_{idx + 1}_epoch_{epoch + 1}_val_acc_{val_accuracy:.4f}.pth")
    torch.save(model_state_dict, save_path)
    trainingLogger.info(f"Best Epoch {idx + 1} saved: Epoch {epoch + 1} with Validation Accuracy: {val_accuracy:.4f}")
##########################################################################################################################
#Final model save (optional)
final_model_path = os.path.join(save_dir, "final_model.pth")
torch.save(model.state_dict(), final_model_path)
trainingLogger.info(f"Final model saved at {final_model_path}")

# Validation loop
model.eval()
correct_predictions = 0
with torch.no_grad():
    for val_images, val_labels in val_loader:
        val_images, val_labels = val_images.to(device), val_labels.to(device)
        val_images = val_images.contiguous()  # Ensure the tensor is contiguous
        outputs = model(val_images)
        _, predicted = torch.max(outputs, 1)
        correct_predictions += (predicted == val_labels).sum().item()

    # Clear GPU memory after validation
    torch.cuda.empty_cache()

    val_accuracy = correct_predictions / len(val_loader.dataset)



############################
#test_loop
import torch
from sklearn.metrics import roc_auc_score
save_dir = "./saved_models_bestTrial"
final_model_path = os.path.join(save_dir, "best_epoch_1_epoch_25_val_acc_0.6111.pth")
# Load the trained model
model.load_state_dict(torch.load(final_model_path))  
model.to(device)  # Move to GPU if available
model.eval()  # Set to evaluation mode

# Testing loop
correct_predictions = 0
all_labels = []
all_probs = []

with torch.no_grad():
    for test_images, test_labels in test_loader:
        test_images, test_labels = test_images.to(device), test_labels.to(device)
        test_images = test_images.contiguous()  # Ensure the tensor is contiguous
        outputs = model(test_images)
        _, predicted = torch.max(outputs, 1)
        
        correct_predictions += (predicted == test_labels).sum().item()
        
        # Collect true labels and predicted probabilities for AUC
        all_labels.extend(test_labels.cpu().numpy())
        all_probs.extend(torch.softmax(outputs, dim=1)[:, 1].cpu().numpy())  # Prob of class 1

    # Compute accuracy
    test_accuracy = correct_predictions / len(test_loader.dataset)

    # Compute AUC (only for binary classification)
    if len(set(all_labels)) == 2:  # Ensure binary classification
         auc_score = roc_auc_score(all_labels, all_probs)
    else:
         auc_score = None  # AUC is not defined for multi-class without one-vs-all strategy

    # Clear GPU memory after testing
    torch.cuda.empty_cache()

# Print results
trainingLogger.info(f"{final_model_path}: Test Accuracy: {test_accuracy:.4f}")
if auc_score is not None:
     trainingLogger.info(f"Test AUC: {auc_score:.4f}")
else:
     trainingLogger.info("AUC not computed (requires binary classification).")
