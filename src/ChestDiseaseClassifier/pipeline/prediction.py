import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os

class PredictionPipeline:
    def __init__(self, filename):
        self.filename = filename
        # Load model once when the class is created
        self.model = load_model(os.path.join("model", "model.h5"), compile=False)
        self.classes = ["Normal", "Adenocarcinoma Cancer"]

    def predict(self):
        # Load and preprocess image
        test_image = image.load_img(self.filename, target_size=(224, 224))
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis=0)
        test_image = test_image / 255.0  # normalize

        # Predict
        result = np.argmax(self.model.predict(test_image), axis=1)
        prediction = self.classes[result[0]]

        # Return in dict format
        return [{"image": prediction}]
