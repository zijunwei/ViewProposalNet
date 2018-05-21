from datasets import custom_transforms
from torchvision import transforms


def get_val_transform(image_size=320):
    transform = transforms.Compose([
                custom_transforms.Resize(w=image_size, h=image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[1.0, 1.0, 1.0])
            ])
    return transform


