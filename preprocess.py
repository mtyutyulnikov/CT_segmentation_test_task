from monai.transforms import (
    EnsureChannelFirstd,
    Compose,
    CropForegroundd,
    Orientationd,
    RandFlipd,
    RandCropByPosNegLabeld,
    RandShiftIntensityd,
    ScaleIntensityRanged,
    Spacingd,
    RandRotate90d,
    SpatialPadd,
    AsDiscreted,
)

roi_size = (48, 96, 96) 

def get_train_transforms(num_samples=1):
    train_transforms = Compose(
        [
            EnsureChannelFirstd(keys=["image", "label"]),
            ScaleIntensityRanged(
                keys=["image"],
                a_min=3071,  # min/max values of dataset images
                a_max=-1024,
                b_min=0.0,
                b_max=1.0,
                clip=True,
            ),
            AsDiscreted(keys=["label"], threshold=0.5),
            CropForegroundd(keys=["image", "label"], source_key="image"),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            Spacingd(
                keys=["image", "label"],
                pixdim=(1.0, 2.0, 2.0),
                mode=("bilinear", "nearest"),
            ),
            SpatialPadd(keys=["image", "label"], spatial_size=roi_size),
            RandCropByPosNegLabeld(
                keys=["image", "label"],
                label_key="label",
                spatial_size=roi_size,
                pos=1,
                neg=2,
                num_samples=num_samples,  
                image_key="image",
                image_threshold=0,
            ),
            RandFlipd(
                keys=["image", "label"],
                spatial_axis=[0],
                prob=0.50,
            ),
            RandFlipd(
                keys=["image", "label"],
                spatial_axis=[1],
                prob=0.50,
            ),
            RandFlipd(
                keys=["image", "label"],
                spatial_axis=[2],
                prob=0.50,
            ),
            RandRotate90d(
                keys=["image", "label"], prob=0.50, max_k=3, spatial_axes=(1, 2)
            ),
            RandShiftIntensityd(
                keys=["image"],
                offsets=0.10,
                prob=0.50,
            ),
        ]
    )
    return train_transforms


def get_val_transforms():
    val_transforms = Compose(
        [
            EnsureChannelFirstd(keys=["image", "label"]),
            ScaleIntensityRanged(
                keys=["image"],
                a_min=3071,  # min/max values of dataset images
                a_max=-1024,
                b_min=0.0,
                b_max=1.0,
                clip=True,
            ),
            AsDiscreted(keys=["label"], threshold=0.5),
            CropForegroundd(keys=["image", "label"], source_key="image"),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            Spacingd(
                keys=["image", "label"],
                pixdim=(1.0, 2.0, 2.0),
                mode=("bilinear", "nearest"),
            ),
            SpatialPadd(keys=["image", "label"], spatial_size=roi_size),
        ]
    )
    return val_transforms
