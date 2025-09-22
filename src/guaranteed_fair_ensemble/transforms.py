import torchvision.transforms.v2 as transforms

from guaranteed_fair_ensemble.backbone import WEIGHTS_DICT


def get_train_transforms():
    """
    Get standard transforms for training images
    """
    return [
        transforms.RandomRotation(degrees=15),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(),
    ]


def get_transforms(is_train=True, backbone_name: str = "mobilenetv3"):
    """
    Factory function to get appropriate transforms based on method and training phase

    Args:
        is_train: Whether transforms are for training (True) or testing (False)
        backbone_name: Name of the backbone model to determine specific transforms


    Returns:
        Appropriate transforms composition
    """
    backbone_transforms = WEIGHTS_DICT[backbone_name].transforms()

    return (
        transforms.Compose([*get_train_transforms(), backbone_transforms])
        if is_train
        else backbone_transforms
    )
