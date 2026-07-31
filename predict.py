# ==========================
# Import Required Libraries
# ==========================

import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image


# Load Trained Model


model = load_model("model.keras")

# ==========================
# Class Names
# ==========================

classes = [
    "Adenocarcinoma",
    "Normal Lung",
    "Squamous Cell Carcinoma"
]

# ==========================
# Prediction Function
# ==========================

def predict_lung_image(img_path):

    img = image.load_img(img_path, target_size=(128, 128))

    img_array = image.img_to_array(img)

    img_array = img_array / 255.0

    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)

    predicted_index = np.argmax(prediction)

    confidence = float(np.max(prediction) * 100)

    predicted_class = classes[predicted_index]

    return predicted_class, confidence