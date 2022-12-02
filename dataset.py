import SimpleITK as sitk
import numpy as np
from pytorch_lightning import LightningDataModule
from torch.utils.data import Dataset, DataLoader, Subset
from pathlib import Path
import nibabel as nib
import scipy
from preprocess import get_train_transforms, get_val_transforms
from sklearn.model_selection import train_test_split
import monai


def load_dicom(directory):
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(directory)
    reader.SetFileNames(dicom_names)
    image_itk = reader.Execute()

    image_zyx = sitk.GetArrayFromImage(image_itk).astype(np.int16)
    return image_zyx


def load_mask(directory):
    mask = nib.load(directory)
    mask = mask.get_fdata().transpose(2, 0, 1)
    mask = scipy.ndimage.rotate(mask, 90, (1, 2))
    return mask


class MedicalDataset(Dataset):
    def __init__(self, imgs_data_path, masks_data_path, transforms=None):
        self.imgs_data_path = Path(imgs_data_path)
        self.imgs_folders = sorted(
            list(filter(lambda p: p.is_dir(), self.imgs_data_path.iterdir()))
        )

        self.masks_data_path = Path(masks_data_path)
        self.masks_folders = sorted(
            list(filter(lambda p: p.is_dir(), self.masks_data_path.iterdir()))
        )
        self.transforms = transforms

        assert len(self.imgs_folders) == len(self.masks_folders)
        for i, j in zip(self.imgs_folders, self.masks_folders):
            assert i.name == j.name

    def __len__(self):
        return len(self.imgs_folders)

    def __getitem__(self, index):
        img_folder = self.imgs_folders[index]
        mask_folder = self.masks_folders[index]

        img_folder_to_open = next(img_folder.glob("1*/*/1*/*")).parent
        img = load_dicom(str(img_folder_to_open))

        mask_file = next(mask_folder.glob("*.nii.gz"))
        mask = load_mask(str(mask_file))
        

        sample = {"image": img, "label": mask}
        if self.transforms is not None:
            sample = self.transforms(sample)  # list of dicts

        return sample


class MedicalDataModule(LightningDataModule):
    def __init__(self, imgs_dir, masks_dir, imgs_num=1, samples_per_img=1, num_workers=8):
        super().__init__()
        self.imgs_dir = imgs_dir
        self.masks_dir = masks_dir
        self.imgs_num = imgs_num
        self.samples_per_img = samples_per_img
        self.num_workers = num_workers

    def setup(self, stage=None):
        train_transformed_dataset = MedicalDataset(
            self.imgs_dir,
            self.masks_dir,
            transforms=get_train_transforms(num_samples=self.samples_per_img),
        )
        val_transformed_dataset = MedicalDataset(
            self.imgs_dir,
            self.masks_dir,
            transforms=get_val_transforms(),
        )

        train_indices, val_indices = train_test_split(
            list(range(len(train_transformed_dataset))), test_size=2
        )

        self.train_dataset = Subset(train_transformed_dataset, train_indices)
        self.val_dataset = Subset(val_transformed_dataset, val_indices)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.imgs_num,
            num_workers=self.num_workers,
            collate_fn=monai.data.utils.list_data_collate,
        )

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=1, num_workers=self.num_workers)
