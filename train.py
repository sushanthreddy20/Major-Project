import os
import json
import random
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models

# =========================
# PATHS
# =========================
TRAIN_PATH = r"C:\Users\Bunny\OneDrive\Documents\Dataset\Train"
VAL_PATH = r"C:\Users\Bunny\OneDrive\Documents\Dataset\Val"

MODEL_DIR = "models"
OUTPUT_DIR = "outputs"
MODEL_PATH = os.path.join(MODEL_DIR, "model.h5")
HISTORY_PATH = os.path.join(MODEL_DIR, "history.json")
CLASS_NAMES_PATH = os.path.join(MODEL_DIR, "class_names.json")

# =========================
# SETTINGS
# =========================
IMG_SIZE = (128, 128)        # smaller image size = faster
BATCH_SIZE = 8               # smaller batch for quick CPU run
EPOCHS = 2                   # reduce epochs
MAX_IMAGES_PER_CLASS = 20    # use only 20 images per class

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =========================
# VALID IMAGE EXTENSIONS
# =========================
VALID_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".jfif", ".webp")

# =========================
# CREATE SMALL SUBSET FOLDERS
# =========================
SMALL_TRAIN_PATH = "small_dataset/train"
SMALL_VAL_PATH = "small_dataset/val"

def create_small_subset(src_root, dst_root, max_images_per_class):
    os.makedirs(dst_root, exist_ok=True)

    class_names = [
        d for d in os.listdir(src_root)
        if os.path.isdir(os.path.join(src_root, d))
    ]

    if len(class_names) == 0:
        raise Exception(f"No class folders found in: {src_root}")

    for class_name in class_names:
        src_class = os.path.join(src_root, class_name)
        dst_class = os.path.join(dst_root, class_name)
        os.makedirs(dst_class, exist_ok=True)

        images = [
            f for f in os.listdir(src_class)
            if f.lower().endswith(VALID_EXTS)
        ]

        random.shuffle(images)
        selected_images = images[:max_images_per_class]

        for img_name in selected_images:
            src_img = os.path.join(src_class, img_name)
            dst_img = os.path.join(dst_class, img_name)

            if not os.path.exists(dst_img):
                with open(src_img, "rb") as fsrc:
                    with open(dst_img, "wb") as fdst:
                        fdst.write(fsrc.read())

# recreate small dataset only once
if not os.path.exists(SMALL_TRAIN_PATH) or not os.path.exists(SMALL_VAL_PATH):
    create_small_subset(TRAIN_PATH, SMALL_TRAIN_PATH, MAX_IMAGES_PER_CLASS)
    create_small_subset(VAL_PATH, SMALL_VAL_PATH, max(5, MAX_IMAGES_PER_CLASS // 4))

# =========================
# DATA GENERATORS
# =========================
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255.0,
    rotation_range=10,
    zoom_range=0.1,
    horizontal_flip=True
)

val_datagen = ImageDataGenerator(rescale=1.0 / 255.0)

train_data = train_datagen.flow_from_directory(
    SMALL_TRAIN_PATH,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=True
)

val_data = val_datagen.flow_from_directory(
    SMALL_VAL_PATH,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False
)

if train_data.num_classes == 0:
    raise Exception("Training dataset has 0 classes. Check folder structure.")

class_names = list(train_data.class_indices.keys())
with open(CLASS_NAMES_PATH, "w") as f:
    json.dump(class_names, f)

print("Detected classes:", class_names)

# =========================
# FAST MODEL
# =========================
model = models.Sequential([
    layers.Input(shape=(128, 128, 3)),

    layers.Conv2D(16, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(32, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),

    layers.Flatten(),
    layers.Dense(64, activation="relu"),
    layers.Dense(train_data.num_classes, activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS,
    verbose=1
)

model.save(MODEL_PATH)

with open(HISTORY_PATH, "w") as f:
    json.dump(history.history, f)

print("\nTraining completed quickly.")
print(f"Model saved at: {MODEL_PATH}")
print(f"History saved at: {HISTORY_PATH}")
print(f"Class names saved at: {CLASS_NAMES_PATH}")