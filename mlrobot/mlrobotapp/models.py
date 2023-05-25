from django.db import models

# Create your models here.

import tensorflow as tf
from keras import layers
from ssd_resnet import SSDResNet


class tfMask_CRNN(tf.keras.Model):
    def __init__(self, num_classes):
        super(tfMask_CRNN, self).__init__()
        self.num_classes = num_classes

        # Define your model architecture
        self.conv1 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')
        self.batchnorm1 = layers.BatchNormalization()
        self.maxpool1 = layers.MaxPooling2D((2, 2))
        self.conv2 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')
        self.batchnorm2 = layers.BatchNormalization()
        self.maxpool2 = layers.MaxPooling2D((2, 2))
        self.flatten = layers.Flatten()
        self.dropout = layers.Dropout(0.5)
        self.fc1 = layers.Dense(128, activation='relu')
        self.fc2 = layers.Dense(self.num_classes, activation='softmax')

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.batchnorm1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.batchnorm2(x)
        x = self.maxpool2(x)
        x = self.flatten(x)
        x = self.dropout(x)
        x = self.fc1(x)
        x = self.fc2(x)
        return x
    


class SSDResNetModel:
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
