import os
import sys
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image

MODEL_PATH = os.path.join("models", "model.h5")
CLASS_NAMES_PATH = os.path.join("models", "class_names.json")
IMG_SIZE = (128, 128)

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

if not os.path.exists(CLASS_NAMES_PATH):
    raise FileNotFoundError(f"Class names file not found: {CLASS_NAMES_PATH}")

model = tf.keras.models.load_model(MODEL_PATH)

with open(CLASS_NAMES_PATH, "r") as f:
    class_names = json.load(f)

def predict_stage(img_path):
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"Image not found: {img_path}")

    img = image.load_img(img_path, target_size=IMG_SIZE)
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array, verbose=0)[0]
    predicted_index = int(np.argmax(prediction))
    predicted_label = class_names[predicted_index]
    confidence = float(np.max(prediction))

    print("\nPrediction Result")
    print("-----------------")
    print("Image Path       :", img_path)
    print("Predicted Class  :", predicted_label)
    print("Confidence       :", round(confidence * 100, 2), "%")

    print("\nAll Class Probabilities:")
    for i, prob in enumerate(prediction):
        print(f"{class_names[i]} : {prob * 100:.2f}%")

if __name__ == "__main__":
    if len(sys.argv) >= 2:
        img_path = sys.argv[1]
    else:
        img_path = input("Enter full image path: ").strip()

    predict_stage(img_path)