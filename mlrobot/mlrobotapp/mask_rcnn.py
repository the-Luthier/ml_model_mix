import configparser
import numpy as np
import os
import sys
import tensorflow as tf
from mrcnn.config import Config
from mrcnn import model as modellib
from mrcnn import utils
from mrcnn import visualize


class CustomConfig(configparser.ConfigParser):
    # Customize the necessary configuration parameters
    NAME = "custom_config"
    IMAGES_PER_GPU = 1
    # ...

    def __init__(self, num_classes):
        super().__init__()
        self.NUM_CLASSES = 1 + num_classes  # Number of classes in your dataset

        # Load default settings
        self.load_defaults()

    def load_defaults(self):
        # Set default values
        self["DEFAULT"] = {
            "NAME": self.NAME,
            "IMAGES_PER_GPU": str(self.IMAGES_PER_GPU),
            "NUM_CLASSES": str(self.NUM_CLASSES),
            # Add other default parameters here
        }

    def save(self, filepath):
        with open(filepath, "w") as file:
            self.write(file)

    def load(self, filepath):
        self.read(filepath)


# Example usage
num_classes = 10  # Replace with the actual number of classes in your dataset
config = CustomConfig(num_classes)
config.save("custom_config.ini")

# Later, you can load the configuration from the saved file
loaded_config = CustomConfig(num_classes)
loaded_config.load("custom_config.ini")

# Access the configuration parameters
name = loaded_config.get("DEFAULT", "NAME")
images_per_gpu = loaded_config.getint("DEFAULT", "IMAGES_PER_GPU")
num_classes = loaded_config.getint("DEFAULT", "NUM_CLASSES")
# ...
