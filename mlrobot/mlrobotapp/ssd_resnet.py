import torch
import torchvision
from torchvision.models.detection import ssdlite320_mobilenet_v3_large


class SSDResNet:
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def build_model(self):
        # Load the pre-trained SSD ResNet-152 model
        self.model = ssdlite320_mobilenet_v3_large(pretrained=True)

        # Replace the classification head with a new one suitable for the custom number of classes
        num_inputs = self.model.classification[-1].in_features
        self.model.classification[-1] = torch.nn.Linear(num_inputs, self.num_classes + 1)

        # Move the model to the device (GPU if available)
        self.model = self.model.to(self.device)

    def get_model(self):
        return self.model

    def get_device(self):
        return self.device
