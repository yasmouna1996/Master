import torch
import monai
from monai.transforms import (
    Compose, LoadImage, ToTensor, NormalizeIntensity, Resize
)
from monai.networks.nets import ResNet
#from monai.networks.nets.resnet import BasicBlock, Bottleneck
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import nibabel as nib



class NiftiDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None, edge_detection=False):
        self.image_paths = image_paths  # List of paths to NIfTI files
        self.labels = labels  # List of corresponding labels
        self.transform = transform  # Transformations to apply
        self.edge_detection = edge_detection  # Flag to apply edge detection

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load the NIfTI image and label
        image_path = self.image_paths[idx]
        label = self.labels[idx]

        # Load the NIfTI image using nibabel
        nii_image = nib.load(image_path)

        # Get the image data as a NumPy array
        image = nii_image.get_fdata()

        image = self.extract_edges(image)

        # Add an additional channel dimension (for 3D image, it's already 3D)
        image = np.expand_dims(image, axis=0)

        # Apply transformations like resizing
        if self.transform:
            image = self.transform(image)

        return torch.tensor(image, dtype=torch.float32), torch.tensor(label, dtype=torch.long)

    def extract_edges(self, image):
        edge_thickness = 40
        width = image.shape[1]
        extracted = image[ :edge_thickness, edge_thickness:width,:]
        extracted = np.append( extracted, image[ edge_thickness:width, width- edge_thickness:, :].transpose (1,0,2), axis=1 )
        extracted = np.append( extracted, image[ width-edge_thickness:,0: width- edge_thickness, :], axis=1 )
        extracted = np.append( extracted, image[:width- edge_thickness, :edge_thickness, :].transpose (1,0,2), axis=1 )

        return extracted

# Load the Excel file
excel_file = r"W:\strahlenklinik\science\AI_Radiation\Yasmine\Dataset_Labels.xlsx"
df = pd.read_excel(excel_file)

# Path to your NIfTI files
dataset_dir = Path(r"C:\Users\rkikye\Code\Final_Data")

# Get all NIfTI image paths (assuming they are named with the patient name)
image_paths = list(dataset_dir.glob("*.nii*"))  # This will match both .nii and .nii.gz files

# Extract the patient names from the image filenames
image_patient_names = [image_path.stem + '.gz' for image_path in image_paths]  # Fix the name extraction

# Load the labels from the Excel file
labels_df = pd.read_excel(excel_file)
labels_df.columns = ['PatientName', 'Class', 'Gradient Coil']
labels_df['Class'] = labels_df['Class'].replace({'ND': 1, '2D': 0})

# Filter out images that don't have a label (optional, in case some are missing in the Excel file)
valid_images = [(image_path, label) for image_path, label in zip(image_paths, labels_df['Class'])]
valid_image_paths, valid_labels = zip(*valid_images)

# Shuffle and split the data into training and validation sets (80% train, 20% validation)
train_image_paths, val_image_paths, train_labels, val_labels = train_test_split(
    valid_image_paths, valid_labels, test_size=0.2, random_state=42
)

# Convert lists back to Path objects
train_image_paths = list(train_image_paths)
val_image_paths = list(val_image_paths)

# Print the number of images in each set
print(f"Training set size: {len(train_image_paths)}")
print(f"Validation set size: {len(val_image_paths)}")

# Set device (GPU if available, otherwise CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the image transformations
transform = Compose([
    Resize(spatial_size=(240, 240, 150)),  # Resize to a consistent shape (example size)
])

# Create the dataset and dataloaders with edge detection
train_dataset = NiftiDataset(image_paths=train_image_paths, labels=train_labels, transform=None, edge_detection=True)
train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)

# Load the 3D ResNet model (resnet18 in this case)
block = 'basic'  # Use BasicBlock for ResNet-18
layers = [2, 2, 2, 2]  # ResNet-18
block_inplanes = [64, 128, 256, 512]  # Default starting number of input planes in ResNet

# Initialize the ResNet model
model = ResNet(
    spatial_dims=3,  # For 3D input, use 3
    n_input_channels=1,   # If your input images are grayscale, use 1
    num_classes=2,  # For binary classification or 2 output classes
    block=block,
    layers=layers,
    block_inplanes=block_inplanes
)
model.to(device)  # Move model to GPU or CPU

# For classification, use CrossEntropyLoss
loss_function = torch.nn.CrossEntropyLoss()

# Use Adam optimizer for training
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# Training loop
epochs = 20
for epoch in range(epochs):
    model.train()  # Set the model to training mode
    running_loss = 0.0
    correct_predictions = 0
    total_predictions = 0
    
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()  # Clear the gradients
        
        outputs = model(images)  # Forward pass
        loss = loss_function(outputs, labels)  # Compute loss
        loss.backward()  # Backpropagation
        optimizer.step()  # Update model parameters
        
        running_loss += loss.item()  # Keep track of the loss
        
        # Calculate accuracy
        _, predicted = torch.max(outputs, 1)
        total_predictions += labels.size(0)
        correct_predictions += (predicted == labels).sum().item()
    
    epoch_loss = running_loss / len(train_loader)
    accuracy = correct_predictions / total_predictions
    print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}, Accuracy: {accuracy:.4f}")
