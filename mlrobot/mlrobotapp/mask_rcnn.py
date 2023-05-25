class CustomConfig(Config):
    # Customize the necessary configuration parameters
    NAME = "custom_config"
    IMAGES_PER_GPU = 1
    NUM_CLASSES = 1 + num_classes  # Number of classes in your dataset
    # ...
