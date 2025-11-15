
from rest_framework import status
from django.http import JsonResponse
from rest_framework.views import APIView
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image

class_labels = {
    0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G' }


# Create your views here.

model = tf.keras.models.load_model('predict/guitar_chord_cnnbestoo_model.keras')

class PredictView(APIView):
    def get(self, request):
        return JsonResponse({"message": "Predict API aktif! Gunakan POST untuk kirim gambar."})

    def post(self, request):
        try:
            file = request.FILES.get('image')
            if not file:
                return JsonResponse({"error": "File tidak ada"}, status=status.HTTP_400_BAD_REQUEST)
            
            img = Image.open(file)
            img.save('debug_image.jpg')
            img = img.resize((200, 200))
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0) / 255.0
            

            preds = model.predict(img_array)
            predicted_class_index = np.argmax(preds[0])
            predicted_label = class_labels[predicted_class_index]
            confidence = float(np.max(preds[0]) * 100)
            print("===== DEBUG MODEL OUTPUT =====")
            print("Predictions:", preds)
            print("Predicted index:", predicted_class_index)
            print("Predicted label:", predicted_label)
            print("Confidence:", confidence)
            print("==============================")

            return JsonResponse({
                "predicted_label": predicted_label,
                "confidence": f"{confidence:.2f}%",
                "probabilities": preds.tolist()
            })
        except Exception as e:
            return JsonResponse({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
