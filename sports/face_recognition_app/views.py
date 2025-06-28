from django.shortcuts import render
from django.conf import settings
from django.core.files.storage import FileSystemStorage
from .util import load_saved_artifacts, classify_image
import os

# Load model/assets once
load_saved_artifacts()

def home(request):
    prediction = None
    file_url = None

    if request.method == 'POST' and request.FILES.get('image'):
        print("Image received:", request.FILES['image'].name)
        img = request.FILES['image']

        # Save to media/uploads/
        fs = FileSystemStorage(location=os.path.join(settings.MEDIA_ROOT, 'uploads'))
        filename = fs.save(img.name, img)

        # Correct file_url
        file_url = settings.MEDIA_URL + 'uploads/' + filename
        file_path = fs.path(filename)

        prediction = classify_image(file_path)

        try:
            os.remove(file_path)
        except:
            pass

    return render(request, 'home.html', {
        'prediction': prediction,
        'file_url': file_url
    })
