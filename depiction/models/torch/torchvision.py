"""Core module for torchvision models."""
import torch
import torchvision.transforms as transforms

from .core import TorchModel

# NOTE: From https://pytorch.org/docs/stable/torchvision/models.html.
NORMALIZE = transforms.Normalize(
    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
)


class TorchVisionModel(TorchModel):
    """torchvision model wrapper."""

    def __init__(self, model, task, data_type):
        """
        Initialize a TorchVisionModel.

        Args:
            model (torch.nn.Module): model to wrap.
            task (depiction.core.Task): task type.
            data_type (depiction.core.DataType): data type.
        """
        super().__init__(model=model, task=task, data_type=data_type)

    def _prepare_sample(self, sample):
        """
        Prepare sample for the model.

        Args:
            sample (np.ndarray): an input sample for the model.

        Returns:
            torch.tensor: a tensor representing the sample.
        """
        return torch.stack(
            [
                NORMALIZE(example) for example in
                torch.unbind(TorchModel._prepare_sample(self, sample), dim=0)
            ],
            axis=0
        )
