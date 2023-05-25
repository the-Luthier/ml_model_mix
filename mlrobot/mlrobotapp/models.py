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

    def build_model(self):
        model = SSDResNet(self.num_classes)
        model.build_model()
        self.model = model.get_model()

    def get_model(self):
        return self.model
