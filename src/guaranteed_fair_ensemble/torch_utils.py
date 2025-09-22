import torch
import torch.nn.functional as F


def custom_one_hot(x: torch.Tensor, num_classes: int = 3) -> torch.Tensor:
    """
    Convert input tensor to one-hot encoding where num_classes equals number of unique values
    """
    if num_classes == 2:
        return x.unsqueeze(-1).float()

    # One-hot encode with dynamically determined number of classes
    one_hot = F.one_hot(x.to(torch.int64), num_classes=num_classes)

    return one_hot.float()  # Convert to float for gradient computation


def reverse_one_hot(x: torch.Tensor) -> torch.Tensor:
    """
    Convert one-hot encoded tensor back to original integer values
    """
    if x.shape[1] == 2:
        return x[:, 0]
    # Get the index of the maximum value in each row
    _, indices = torch.max(x, dim=1)
    return indices


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
