import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image

TRAIN_PATH = "small_dataset/train"
MODEL_PATH = os.path.join("models", "model.h5")
OUTPUT_DIR = "outputs"
FEATURES_PATH = os.path.join(OUTPUT_DIR, "features.csv")

os.makedirs(OUTPUT_DIR, exist_ok=True)

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

if not os.path.exists(TRAIN_PATH):
    raise FileNotFoundError(f"Train path not found: {TRAIN_PATH}")

model = tf.keras.models.load_model(MODEL_PATH)

# Build/call model once so input tensors are defined
dummy_input = np.zeros((1, 128, 128, 3), dtype=np.float32)
_ = model(dummy_input, training=False)

# Use second-last layer as feature output
feature_model = Model(
    inputs=model.inputs,
    outputs=model.layers[-2].output
)

valid_exts = (".jpg", ".jpeg", ".png", ".bmp", ".jfif", ".webp")

features = []
labels = []
file_paths = []

class_folders = [d for d in os.listdir(TRAIN_PATH) if os.path.isdir(os.path.join(TRAIN_PATH, d))]

if len(class_folders) == 0:
    raise Exception("No class folders found in training path.")

for class_name in class_folders:
    class_path = os.path.join(TRAIN_PATH, class_name)

    for img_name in os.listdir(class_path):
        if not img_name.lower().endswith(valid_exts):
            continue

        img_path = os.path.join(class_path, img_name)

        try:
            img = image.load_img(img_path, target_size=(128, 128))
            img_array = image.img_to_array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            feat = feature_model.predict(img_array, verbose=0)[0]

            features.append(feat)
            labels.append(class_name)
            file_paths.append(img_path)

        except Exception as e:
            print(f"Skipping {img_path}: {e}")

if len(features) == 0:
    raise Exception("No features extracted. Check image files.")

df = pd.DataFrame(features)
df["label"] = labels
df["file_path"] = file_paths

df.to_csv(FEATURES_PATH, index=False)

print(f"Feature extraction completed.")
print(f"Features saved at: {FEATURES_PATH}")