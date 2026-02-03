
import pickle
import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
import pandas as pd

class OreDataset(Dataset):
    def __init__(self, pickle_path, target_columns=None, transform=None):
        """
        Args:
            pickle_path (str): Path to the pickle file.
            target_columns (list): List of column names to predict.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        with open(pickle_path, 'rb') as f:
            data = pickle.load(f)

        self.grade_df = data[1]
        self.grade_df = data[1]
        images_raw_sources = data[2]
        
        self.low_energy_imgs = []
        self.high_energy_imgs = []
        
        # images_raw_sources contains 5 elements (sources)
        # Each element has [low_energy_list, high_energy_list]
        for source_imgs in images_raw_sources:
            self.low_energy_imgs.extend(images_raw_sources[source_imgs][0])
            self.high_energy_imgs.extend(images_raw_sources[source_imgs][1])
        
        # Ensure alignment
        assert len(self.grade_df) == len(self.low_energy_imgs) == len(self.high_energy_imgs), \
            "Mismatch in data lengths between grades and images."

        if target_columns is None:
            self.target_columns = ["Fe_grade", "Zn_grade", "Pb_grade", "S_grade"]
        else:
            self.target_columns = target_columns

        # Verify columns exist
        for col in self.target_columns:
            if col not in self.grade_df.columns:
                raise ValueError(f"Target column {col} not found in dataframe.")

        self.transform = transform

    def __len__(self):
        return len(self.grade_df)

    def __getitem__(self, idx):
        # Retrieve images
        # Ensure we're accessing by integer index if dataframe index is not default
        # But images are lists, so idx works for them.
        # Dataframe might have arbitrary index, so use iloc.
        
        low_img_arr = self.low_energy_imgs[idx]
        high_img_arr = self.high_energy_imgs[idx]
        
        # Convert to PIL Image for transforms (Data is likely numpy array or similar)
        # Assuming arrays are in 0-255 range.
        low_img = Image.fromarray(np.uint8(low_img_arr))
        high_img = Image.fromarray(np.uint8(high_img_arr))
        
        # Apply transforms if any
        # Note: We need to apply the SAME random transform to both if using random augmentation.
        # For now, assuming deterministic resize/tensor conversion or handled in composed transform.
        # If transform is a dictionary or custom dual-transform, handle it. 
        # But standard torchvision transforms work on single images.
        # We handle simple Resize + ToTensor here manually or expect a specific pipeline.
        
        if self.transform:
            # If transform is designed to take both, pass both (custom). 
            # If it's standard torchvision, we might need to stack first?
            # Easier approach: Transform individually if deterministic, or stack then transform.
            # Let's stack them into a 2-channel image first if possible, or transform individually.
            # But PIL doesn't natively support 2-channel well for all ops.
            # Best: Resize both, then ToTensor, then Stack.
            
            # For simplicity in this project phase, let's assume 'transform' handles the PIL -> Tensor flow
            # We will separate Resize vs Totensor in main script or here.
            
            low_img = self.transform(low_img)
            high_img = self.transform(high_img)
            
        # Ensure they are tensors now
        if not isinstance(low_img, torch.Tensor):
            # Fallback if transform didn't convert
            import torchvision.transforms.functional as TF
            low_img = TF.to_tensor(low_img)
            high_img = TF.to_tensor(high_img)

        # Stack: (C, H, W) -> (2, H, W)
        # low_img is (1, H, W) or (H, W). TF.to_tensor makes it (1, H, W) for grayscale.
        
        image_tensor = torch.cat([low_img, high_img], dim=0) 

        # Targets
        targets = self.grade_df.iloc[idx][self.target_columns].values.astype(np.float32)
        target_tensor = torch.tensor(targets)

        return image_tensor, target_tensor, torch.tensor(self.grade_df.iloc[idx]['weight'], dtype=torch.float32)
