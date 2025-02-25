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
from torch.cuda.amp import autocast, GradScaler
import matplotlib.pyplot as plt 
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
        resample = tio.Resample((4,4,4))
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

# Define the Optuna objective function
def objective(trial, train_image_paths, train_labels, val_image_paths, val_labels, device):
    try:
        # Hyperparameter suggestions
        lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
        batch_size = 2
        dropout_rate = trial.suggest_float("dropout", 0.1, 0.5)
        noise_prob = trial.suggest_float("noise_prob", 0.1, 0.5)
        flip_prob = trial.suggest_float("flip_prob", 0.1, 0.5)

        trainingLogger.info(f'trial: {trial}')   
        trainingLogger.info(f'Learning Rate: {lr}')
        trainingLogger.info(f'batch size: {batch_size}')
        trainingLogger.info(f'dropout rate: {dropout_rate}')
        trainingLogger.info(f'noise prob:{noise_prob}')
        trainingLogger.info(f'flip prob:{flip_prob}')

        # Data augmentation transforms
        train_transform = tio.Compose([
            tio.RescaleIntensity(percentiles=(0.5, 99.5)),
            tio.ZNormalization(),
            #tio.Resample((512, 512, 512)),
            #tio.CropOrPad((512, 512, 512)),
            #tio.Resample((2, 2, 2)),
            #tio.RandomNoise(std=0.2, p=noise_prob),
            tio.RandomFlip(flip_probability=flip_prob)
        ])

        val_transform = tio.Compose([
            tio.RescaleIntensity(percentiles=(0.5, 99.5)),
            tio.ZNormalization(),
            #tio.CropOrPad((512, 512, 512)),
            #tio.Resample((2, 2, 2))
        ])

        # Datasets and DataLoaders
        train_dataset = NiftiDataset(train_image_paths, train_labels, transform=train_transform)
        val_dataset = NiftiDataset(val_image_paths, val_labels, transform=val_transform)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        # Model
        model = ResNet(
            spatial_dims=3,
            n_input_channels=1,
            num_classes=2,
            block='basic',
            layers=[2, 2, 2, 2],
            block_inplanes=[64, 128, 256, 512],
        ).to(device)

        # Loss and optimizer
        loss_function = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        # AMP: Use GradScaler
        scaler = GradScaler()

        # Training loop
        for epoch in range(5):  # Fewer epochs for faster trials
            trainingLogger.info(f"current epoch:{epoch}")
            model.train()
            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)
                images = images.contiguous()  # Ensure the tensor is contiguous
                # Debugging: Print or assert tensor shapes
                #print(f"Shape of images before passing to model: {images.shape}")
                assert images.dim() == 5, f"Expected 5 dimensions, got {images.dim()}"
                optimizer.zero_grad()


                # Use AMP for forward + backward
                with autocast(dtype=torch.float16):  # Enables mixed precision
                    outputs = model(images)
                    loss = loss_function(outputs, labels)
        

                # Scale loss and backpropagate
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

            torch.cuda.empty_cache()  # Free GPU memory

        # Validation loop
        model.eval()
        correct_predictions = 0
        with torch.no_grad():
            for val_images, val_labels in val_loader:
                val_images, val_labels = val_images.to(device), val_labels.to(device)
                val_images = val_images.contiguous()  # Ensure the tensor is contiguous


                with autocast(dtype=torch.float16):  # Use AMP for inference
                    outputs = model(val_images)
                    _, predicted = torch.max(outputs, 1)
                    correct_predictions += (predicted == val_labels).sum().item()

            # Clear GPU memory after validation
            torch.cuda.empty_cache()

        accuracy = correct_predictions / len(val_loader.dataset)
        # Report intermediate results for pruning
        trial.report(accuracy, epoch)

        # Prune if the trial is not promising
        if trial.should_prune():
            trainingLogger.info("trial is not promising going to prune")
            raise optuna.exceptions.TrialPruned()
        trainingLogger.info(f'training accuracy: {accuracy}')


        return accuracy
    except RuntimeError as e:
        if "out of memory" in str(e):  # Catch OOM errors
            torch.cuda.empty_cache()  # Free up memory
            trainingLogger.info("Skipping trial due to OOM error.")
            print("Skipping trial due to OOM error.")
            return None  # Signals Optuna to skip this trial
        else:
            raise  # Re-raise other unexpected exceptions


# Set up Optuna study with a pruner
pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=0, interval_steps=1)

# Run Optuna optimization
study = optuna.create_study(direction="maximize", pruner=pruner)
study.optimize(lambda trial: objective(trial, train_image_paths, train_labels, val_image_paths, val_labels, device), n_trials=50)

trainingLogger.info("Best trial:"+ f"  Value: {study.best_trial.value}"+f"  Params: {study.best_trial.params}")
df = study.trials_dataframe()
df.to_csv("./study_results.csv")

# Best hyperparameters
print("Best trial:")
print(f"  Value: {study.best_trial.value}")
print(f"  Params: {study.best_trial.params}")
