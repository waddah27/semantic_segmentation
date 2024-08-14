from torchvision import transforms
import albumentations as albu
from albumentations.pytorch import ToTensorV2


# Define augmentation pipeline
transform_A = albu.Compose([
    albu.Resize(height=256, width=256, always_apply=True),
    albu.RandomRotate90(p=0.5),
    albu.HorizontalFlip(p=0.5),
    albu.VerticalFlip(p=0.5),
    albu.RandomScale(scale_limit=0.2, p=0.5),
    # albu.RandomCrop(width=256, height=256, p=1.0),  # Adjust size to your requirement
    albu.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, p=1.0),
    ToTensorV2()  # Converts image to tensor and performs normalization
], additional_targets={'mask': 'mask'}, is_check_shapes=False)  # Apply transformations to masks as well

transform = transforms.Compose([
    transforms.Resize((256, 256)),  # Resize to match model input
    transforms.ToTensor()
])
def get_training_augmentation():
    img_shape = (256, 256)
    train_transform = [

        albu.HorizontalFlip(p=0.5),

        albu.ShiftScaleRotate(scale_limit=0.5, rotate_limit=0, shift_limit=0.1, p=1, border_mode=0),

        albu.PadIfNeeded(min_height=img_shape[0], min_width=img_shape[1], always_apply=True, border_mode=0, value=0),
        albu.RandomCrop(height=img_shape[0], width=img_shape[1], always_apply=True),

        albu.GaussNoise(p=0.2),
        albu.Perspective(p=0.5),

        albu.OneOf(
            [
                albu.CLAHE(p=1),
                albu.RandomBrightnessContrast(p=1),
                albu.RandomGamma(p=1),
            ],
            p=0.9,
        ),

        albu.OneOf(
            [
                # albu.IAASharpen(p=1),
                albu.Blur(blur_limit=3, p=1),
                albu.MotionBlur(blur_limit=3, p=1),
            ],
            p=0.9,
        ),

        albu.OneOf(
            [
                albu.RandomBrightnessContrast(p=1),
                albu.HueSaturationValue(p=1),
            ],
            p=0.9,
        ),
        ToTensorV2(),
    ]
    return albu.Compose(train_transform)


def get_validation_augmentation():
    """Add paddings to make image shape divisible by 32"""
    test_transform = [
        albu.PadIfNeeded(384, 480),
        ToTensorV2(),
    ]
    return albu.Compose(test_transform)


def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')


def get_preprocessing(preprocessing_fn):
    """Construct preprocessing transform

    Args:
        preprocessing_fn (callbale): data normalization function
            (can be specific for each pretrained neural network)
    Return:
        transform: albumentations.Compose

    """

    _transform = [
        albu.Lambda(image=preprocessing_fn),
        albu.Lambda(image=to_tensor, mask=to_tensor),
    ]
    return albu.Compose(_transform)