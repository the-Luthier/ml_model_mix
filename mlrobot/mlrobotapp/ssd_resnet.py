import torch
import torchvision
from torchvision.models.detection import ssdlite320_mobilenet_v3_large
from torch.utils.data import DataLoader

class SSDResNet:
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def build_model(self):
        # Load the pre-trained SSD ResNet-152 model
        self.model = ssdlite320_mobilenet_v3_large(pretrained=True)

        # Replace the classification head with a new one suitable for the custom number of classes
        num_inputs = self.model.head.classifier[-1].in_features
        self.model.head.classifier = torch.nn.Linear(num_inputs, self.num_classes + 1)

        # Move the model to the device (GPU if available)
        self.model = self.model.to(self.device)

    def train_model(self, train_dataset, val_dataset, num_epochs, batch_size, learning_rate):
        if self.model is None:
            print("Model not initialized. Call 'build_model' method first.")
            return

        # Create data loaders for training and validation sets
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        # Define loss function and optimizer
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(self.model.parameters(), lr=learning_rate, momentum=0.9)

        # Training loop
        for epoch in range(num_epochs):
            self.model.train()
            train_loss = 0.0

            for images, targets in train_dataloader:
                # Move data to the device
                images = images.to(self.device)
                targets = targets.to(self.device)

                # Zero the gradients
                optimizer.zero_grad()

                # Forward pass
                outputs = self.model(images)

                # Compute loss
                loss = criterion(outputs['logits'], targets)
                train_loss += loss.item()

                # Backward pass and optimization
                loss.backward()
                optimizer.step()

            # Print training loss for the epoch
            avg_train_loss = train_loss / len(train_dataloader)
            print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}")

            # Evaluate the model on the validation set
            self.evaluate_model(val_dataloader)

    def evaluate_model(self, dataloader):
        self.model.eval()
        total_correct = 0
        total_samples = 0

        with torch.no_grad():
            for images, targets in dataloader:
                # Move data to the device
                images = images.to(self.device)
                targets = targets.to(self.device)

                # Forward pass
                outputs = self.model(images)

                # Get predicted class labels
                _, predicted = torch.max(outputs['logits'], 1)

                # Update accuracy statistics
                total_correct += (predicted == targets).sum().item()
                total_samples += targets.size(0)

        # Calculate accuracy
        accuracy = total_correct / total_samples
        print(f"Validation Accuracy: {accuracy:.4f}")

    def get_model(self):
        return self.model

    def get_device(self):
        return self.device
