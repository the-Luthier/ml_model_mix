from django.shortcuts import render

# Create your views here.

import cv2
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from .models import tfMask_CRNN


@csrf_exempt
def inference_tfMask_CRNN(request):
    if request.method == 'POST':
        # Load the incoming image
        image = cv2.imdecode(request.FILES['image'].read(), cv2.IMREAD_COLOR)
        
        # Preprocess the image (e.g., resize, normalization)
        # ...
        
        # Perform inference using the trained model
        num_classes = request(id)
        model = tfMask_CRNN(num_classes)  # Create an instance of your trained model
        output = model.predict(image)
        
        # Process the output (e.g., post-processing, extracting predictions)
        # ...
        
        # Return the results as a JSON response
        return JsonResponse({'results': output})
    else:
        return JsonResponse({'message': 'Invalid request method.'})
