from django.shortcuts import render

# Create your views here.

import cv2
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from rest_framework.views import APIView
from models import tfMask_CRNN
from models import SSDResNetModel
from mask_rcnn import CustomDataset
import torch
from torchvision.transforms import ToTensor 
from ssd_resnet import SSDResNet
import numpy as np

class Mask_CRNNview:
    @csrf_exempt
    def post(self, request):
        # Load the incoming image
        image = cv2.imdecode(request.FILES['image'].read(), cv2.IMREAD_COLOR)

        # Preprocess the image (e.g., resize, normalization)
        # ...

        # Perform inference using the trained model
        num_classes = self.get_num_classes(request)
        model = tfMask_CRNN(num_classes)  # Create an instance of your trained model
        output = model.predict(image)

        # Process the output (e.g., post-processing, extracting predictions)
        # ...

        # Return the results as a JSON response
        return JsonResponse({'results': output})

    def get_num_classes(self, request):
        # Extract the number of classes from the request
        # (e.g., retrieve from the request parameters or headers)
        num_classes = 0  # Replace with the appropriate code to extract the number of classes
        return num_classes


class SSDResNetView:
    def __init__(self):
        self.model = SSDResNetModel(num_classes=10)
        self.model.build_model()

        self.dataset = CustomDataset()
        self.dataset.load_dataset("path/to/dataset")
        self.dataset.prepare()

    def detect_objects(self, request):
        # Get an image from the dataset
        image_id = 1234  # Replace with the actual image ID
        image = self.dataset.load_image(image_id)
        image_tensor = ToTensor()(image)

        # Perform object detection on the image
        detection_results = self.detect_objects_with_model(image_tensor)

        # Pass the detection results to the template
        context = {
            'image': image,
            'detection_results': detection_results,
        }
        return render(request, 'detect_objects.html', context)

    def detect_objects_with_model(self, image):
        # Convert the image to a batch of size 1
        image = image.unsqueeze(0)

        # Make predictions on the image
        self.model.get_model()
        with torch.no_grad():
            predictions = self.new_method(image)

        # Process the predictions and return the results
        detection_results = self.process_predictions(predictions)
        return detection_results

    def new_method(self, image):
        return self.model.get_model()

    def process_predictions(self, predictions):
        # Process the model predictions to extract relevant information
        # such as bounding box coordinates, class labels, and confidence scores
        # Return the detection results in a suitable format
        results = []

        # Example code for processing predictions
        for box, label, score in zip(predictions['boxes'], predictions['labels'], predictions['scores']):
            result = {
                'box': box.item(),
                'label': label.item(),
                'score': score.item(),
            }
            results.append(result)

        return results



class ObjectDetectionView(APIView):
    """
    Detect objects in images.

    Args:
        request: The HTTP request object.

    Returns:
        A JSON response containing the predicted classes.
    """

    def detect_objects(self, request):
        """
        Detect objects in the given image.

        Args:
            request: The HTTP request object.

        Returns:
            The predicted classes as a list.
        """
        image = request.FILES['image']
        model = SSDResNetModel(num_classes=1000)
        device = model.build_model()
        model = model.get_model(device)
        inputs = torch.from_numpy(np.array(image)).float()
        inputs = inputs.unsqueeze(0)
        predictions = model(inputs)
        predictions = predictions.cpu().numpy()
        objects = np.argmax(predictions, axis=1)
        return JsonResponse({'objects': objects})
