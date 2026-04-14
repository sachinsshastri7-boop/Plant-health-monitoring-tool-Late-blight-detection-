import numpy as np
import os
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model

model = load_model("plant_disease_model.h5")

classes = ['Healthy', 'LateBlight']

folder = "test_images"  # create this folder

for file in os.listdir(folder):
    img_path = os.path.join(folder, file)

    img = image.load_img(img_path, target_size=(128,128))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    result = classes[np.argmax(prediction)]

    print(f"{file} → {result}")