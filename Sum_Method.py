import torchio as tio
import os 
from glob import glob
from pathlib import Path
import pandas as pd
import numpy as np


#  Load the Excel file
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
labels_df["voxel_sum"] = 0


def lambda_fn(tensor):
    # set the 0 values to 512
    tensor[tensor == 0] = 512
    return tensor

bambi = tio.Lambda(lambda_fn)
resample = tio.Resample(1)
crop_or_pad = tio.CropOrPad((320,320,320))
resample1 = tio.Resample((2,2,2))

for idx, row in labels_df.iterrows():
    # Get the image path
    image_path = dataset_dir / f"!!{row['PatientName']}"
    # Load the image
    img = tio.ScalarImage(image_path)
    # Get the voxel sum

    img = bambi(img)

    img = resample(img)

    img = crop_or_pad(img)
    
    img = resample1(img)
    print("Shape after second resampling:", img.shape)

    img.save(f"!!{row['PatientName'][:-7]}_preprocessed.nii.gz")

    sum_of_512 = (img.data == 512).sum()
    labels_df.loc[idx, "voxel_sum"] = int(sum_of_512)

labels_df.to_excel("labels_with_voxel_sum.xlsx", index=False)

#Classify the images based on the voxel sum
labels_df["Distortion Correction"] = "ND"   # Default to ND
labels_df.loc[labels_df["voxel_sum"] > 200, "Distortion Correction"] = "2D"  # Set to 2D if voxel sum > 200
labels_df.to_excel("labels_with_voxel_sum_and_classification.xlsx", index=False)        
print("Updated Excel file saved as labels_with_voxel_sum_and_classification.xlsx")
print("Updated Excel file saved as labels_with_voxel_sum.xlsx")
print("Done")


