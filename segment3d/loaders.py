# Ref: https://www.kaggle.com/code/killa92/3d-liver-segmentation-using-pytorch

import json, os, torch, cv2, random
import numpy as np
import nibabel as nib
from matplotlib import pyplot as plt
from glob import glob
from torch.utils.data import random_split, Dataset, DataLoader
from PIL import Image
from torchvision import transforms as tfs
from tqdm import tqdm
import time
import torchio as tio
from matplotlib import animation

# Utility function to ensure that the .temp folder exists
def create_directory_if_not_exists(directory=".temp"):
    """Creates a directory if it doesn't exist."""
    if not os.path.exists(directory):
        os.makedirs(directory)

class CustomSegmentationDataset(Dataset):
    """
    Custom dataset class for segmentation tasks using 3D NIfTI images.
    
    Attributes:
        root (str): Root directory containing image and ground truth segmentations.
        transformations: Albumentations transformations to apply on images and masks.
    """

    def __init__(self, root, transformations=None):
        """
        Initializes the dataset by loading full 3D volumes and ground truth masks.
        
        Args:
            root (str): Root directory containing the images and ground truths.
            transformations: Augmentations to apply on volumes and masks (default: None).
        """
        self.im_nii_paths = sorted(glob(f"{root}/images/*.nii"))
        self.gt_nii_paths = sorted(glob(f"{root}/segmentations/*.nii"))
        self.transformations = transformations
        self.n_cls = 2
        
        assert len(self.im_nii_paths) == len(self.gt_nii_paths), "Mismatch between images and ground truths."

    def __len__(self): 
        """Returns the number of 3D volumes in the dataset."""
        return len(self.im_nii_paths)

    def __getitem__(self, idx):
        """
        Retrieves a single 3D volume and its corresponding ground truth mask at the given index.
        
        Args:
            idx (int): Index of the volume to retrieve.
            
        Returns:
            tuple: 3D volume and ground truth mask.
        """
        im_nii, gt_nii = self.im_nii_paths[idx], self.gt_nii_paths[idx]

        nii_im_data, nii_gt_data = self.read_nii(im_nii, gt_nii)

        if self.transformations: 
            nii_im_data, nii_gt_data = self.apply_transformations(nii_im_data, nii_gt_data)
        
        nii_im_data = self.normalize_image(nii_im_data)
        nii_gt_data[nii_gt_data > 1] = 1  # To avoid issues with cross-entropy loss

        return torch.tensor(nii_im_data).float(), torch.tensor(nii_gt_data).long()

    def normalize_image(self, im): 
        """
        Normalizes the image by setting negative values to 0 and dividing by the maximum pixel value.
        
        Args:
            im (Tensor): Input image to normalize.
            
        Returns:
            Tensor: Normalized image.
        """
        max_val = torch.max(im)
        im[im < 0] = 0
        return im / max_val
    
    # def extract_slices(self, im_nii_paths, gt_nii_paths): 
    #     """
    #     Extracts 2D slices from 3D NIfTI images and ground truth masks.
        
    #     Args:
    #         im_nii_paths (list): List of paths to image NIfTI files.
    #         gt_nii_paths (list): List of paths to ground truth NIfTI files.
            
    #     Returns:
    #         tuple: Lists of image slices and corresponding ground truth masks.
    #     """
    #     ims, gts = [], []
    #     for index, (im_nii, gt_nii) in enumerate(zip(im_nii_paths, gt_nii_paths)):
    #         if index == 50: 
    #             break
    #         nii_im_data, nii_gt_data = self.read_nii(im_nii, gt_nii)
    #         for idx, (im, gt) in enumerate(zip(nii_im_data, nii_gt_data)):
    #             if len(np.unique(gt)) == 2:  # Only consider binary segmentation
    #                 ims.append(im)
    #                 gts.append(gt)
    #     return ims, gts

    def read_nii(self, im, gt): 
        """
        Reads 3D NIfTI volumes and ground truth masks.
        
        Args:
            im (str): Path to the image NIfTI file.
            gt (str): Path to the ground truth NIfTI file.
            
        Returns:
            tuple: 3D image data and ground truth mask.
        """
        return nib.load(im).get_fdata(), nib.load(gt).get_fdata()
    
    def apply_transformations(self, im, gt):
        """
        Applies 3D augmentations to the image and ground truth mask using TorchIO.
        
        Args:
            im (ndarray): Input 3D image (e.g., a NIfTI volume).
            gt (ndarray): Input 3D ground truth mask.
            
        Returns:
            tuple: Transformed image and ground truth mask as Torch tensors.
        """
        # Convert image and mask to TorchIO's Subject format for 3D transformations
        subject = tio.Subject(
            image=tio.ScalarImage(tensor=torch.tensor(im).unsqueeze(0)),  # Add channel dimension
            mask=tio.LabelMap(tensor=torch.tensor(gt).unsqueeze(0))  # Add channel dimension
        )
        
        # Apply the transformations
        transformed_subject = self.transformations(subject)
        
        # Return the transformed image and mask
        transformed_im = transformed_subject['image'].data.squeeze(0)  # Remove channel dimension
        transformed_gt = transformed_subject['mask'].data.squeeze(0)  # Remove channel dimension
        
        return transformed_im, transformed_gt

def create_dataloaders(root, transformations, batch_size, split=[0.9, 0.05, 0.05], num_workers=2):
    """
    Creates dataloaders for training, validation, and test datasets.
    
    Args:
        root (str): Root directory of the dataset.
        transformations: Augmentations to apply to the dataset.
        batch_size (int): Batch size for the dataloaders.
        split (list): Proportion of train, validation, and test sets (default: [0.9, 0.05, 0.05]).
        num_workers (int): Number of workers for data loading (default: 4).
        
    Returns:
        tuple: Dataloaders for training, validation, and test datasets, and number of classes.
    """
    assert sum(split) == 1.0, "Sum of the split must equal 1."
    
    dataset = CustomSegmentationDataset(root=root, transformations=transformations)
    n_classes = dataset.n_cls
    
    # Calculate dataset lengths for train, val, test splits
    train_len = int(len(dataset) * split[0])
    val_len = int(len(dataset) * split[1])
    test_len = len(dataset) - (train_len + val_len)
    
    # Split dataset
    train_ds, val_ds, test_ds = random_split(dataset, [train_len, val_len, test_len])
    
    print(f"\nTraining set: {len(train_ds)} images")
    print(f"Validation set: {len(val_ds)} images")
    print(f"Test set: {len(test_ds)} images\n")
    
    # Create dataloaders
    train_dl = DataLoader(dataset=train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_dl = DataLoader(dataset=val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_dl = DataLoader(dataset=test_ds, batch_size=1, shuffle=False, num_workers=num_workers)
    
    return train_dl, val_dl, test_dl, n_classes

def plot_image(rows, cols, count, im, gt=None, title="Original Image"):
    """
    Plots a single image or mask with a given title.
    
    Args:
        rows (int): Number of rows in the plot grid.
        cols (int): Number of columns in the plot grid.
        count (int): Position of the image in the grid.
        im (Tensor): Image to plot.
        gt (bool): Flag indicating whether the image is a ground truth mask.
        title (str): Title for the plot.
        
    Returns:
        int: Updated position count.
    """
    plt.subplot(rows, cols, count)
    if gt:
        plt.imshow(im.squeeze(0).float())
    else:
        plt.imshow((im * 255).cpu().numpy().astype("uint8") * 255)
    plt.axis("off")
    plt.title(title)
    
    return count + 1

def save_visualization(ds, num_images):
    """
    Visualizes and saves a set of random 2D slices from the 3D volumes and ground truth masks.
    
    Args:
        ds: Dataset to visualize from.
        num_images (int): Number of images to visualize.
    """
    plt.figure(figsize=(25, 20))
    rows = num_images // 4
    cols = num_images // rows
    count = 1
    indices = [random.randint(0, len(ds) - 1) for _ in range(num_images)]
    
    for idx, index in enumerate(indices):
        if count == num_images + 1:
            break
        
        # Load full 3D volumes
        im, gt = ds[index]
        
        # Select a random depth index for slicing (from the first dimension of 3D data)
        depth_idx = random.randint(0, im.shape[0] - 1)

        # Use the selected depth index to extract a 2D slice from the 3D volume
        im_slice = im[depth_idx, :, :]
        gt_slice = gt[depth_idx, :, :]

        # First Plot: Image Slice
        count = plot_image(rows, cols, count, im=im_slice)
        
        # Second Plot: Ground Truth Mask Slice
        count = plot_image(rows, cols, count, im=gt_slice, gt=True, title="Ground Truth Mask")

    # Save the figure with a timestamp
    create_directory_if_not_exists(".temp")
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    plt.savefig(f".temp/visualization_{timestamp}.png")
    print(f"Visualization saved as .temp/visualization_{timestamp}.png")
    plt.close()

def save_animation(ds, image_ind=0):
    """
    Creates and saves an animation that visualizes slices of a 3D volume.
    
    Args:
        ds: Dataset to retrieve the 3D image and mask.
        image_ind (int): Index of the image in the dataset to visualize.
    """
    # Get the 3D image and mask from the dataset
    im, gt = ds[image_ind]
    
    # Set up the figure and axis for the animation
    fig, ax = plt.subplots()
    ims = []  # To store the images for animation
    
    # Loop through the depth of the 3D volume and create an animation of slices
    for depth_idx in range(im.shape[0]):
        im_slice = im[depth_idx, :, :].cpu().numpy()  # Extract a single slice (convert to numpy)
        gt_slice = gt[depth_idx, :, :].cpu().numpy()  # Extract corresponding ground truth mask slice

        # Create an overlay of the image and mask (for visualization)
        im_display = ax.imshow(im_slice, cmap='gray', animated=True)
        gt_display = ax.imshow(gt_slice, cmap='jet', alpha=0.5, animated=True)  # Overlay with transparency
        
        ims.append([im_display, gt_display])  # Append the tuple for each frame

    # Create the animation
    ani = animation.ArtistAnimation(fig, ims, interval=100, blit=True, repeat_delay=1000)

    # Save the animation
    create_directory_if_not_exists(".temp")
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    animation_path = f".temp/volume_animation_{timestamp}.gif"
    
    ani.save(animation_path, writer='pillow')  # Save as GIF using Pillow writer
    print(f"Animation saved as {animation_path}")
    
    plt.close()  # Close the figure to free memory


# Example usage
root = "../datasets/liver3d/"
# Define transformations for 3D images
transformations = tio.Compose([
    tio.Resize((256, 256, 256)),  # Resizing 3D volumes (depth, height, width)
    tio.ToCanonical(),  # Ensure the image orientation is standard
    tio.RescaleIntensity((0, 1)),  # Normalize intensities
    tio.RandomFlip(axes=(0, 1, 2)),  # Randomly flip along depth, height, width
    tio.ZNormalization()  # Z-normalization using mean and std for the whole volume
])

train_dl, val_dl, test_dl, n_classes = create_dataloaders(root=root, transformations=transformations, batch_size=16)

# Save the visualization
save_visualization(train_dl.dataset, num_images=20)

save_animation(train_dl.dataset)


# Test the loaders
print("Train -------------------------------->")
for ind, batch in enumerate(train_dl):
    X, y = batch
    print(X.shape, y.shape)
    pass

print("Val   -------------------------------->")

for ind, batch in enumerate(val_dl):
    X, y = batch
    print(X.shape, y.shape)
    pass

print("Test  -------------------------------->")

for ind, batch in enumerate(test_dl):
    X, y = batch
    print(X.shape, y.shape)
    pass