!pip install opencv-python tensorflow matplotlib scikit-image
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import graycomatrix, graycoprops
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from google.colab import files
uploaded = files.upload()

image_path = list(uploaded.keys())[0]
img = cv2.imread(image_path)
img = cv2.resize(img, (224, 224))
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

plt.imshow(img_rgb)
plt.title("Uploaded Gastric Image")
plt.axis('off')
mean_color = np.mean(img_rgb, axis=(0,1))
print("Average Color (R,G,B):", mean_color)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
_, thresh = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY_INV)

area = np.sum(thresh == 255)
total_pixels = thresh.size

size_percentage = (area / total_pixels) * 100
print("Estimated Tumor Size (% of image):", size_percentage)
glcm = graycomatrix(gray, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)

contrast = graycoprops(glcm, 'contrast')[0,0]
energy = graycoprops(glcm, 'energy')[0,0]
homogeneity = graycoprops(glcm, 'homogeneity')[0,0]

print("Texture Features:")
print("Contrast:", contrast)
print("Energy:", energy)
print("Homogeneity:", homogeneity)
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(224,224,3)),
    MaxPooling2D(2,2),

    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),

    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(2,2),

    Flatten(),
    Dense(128, activation='relu'),
    Dense(4, activation='softmax')  # 4 stages
])
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
# Dummy data (for demonstration)
X_dummy = np.random.rand(20,224,224,3)
y_dummy = np.random.randint(0,4,20)

from tensorflow.keras.utils import to_categorical
y_dummy = to_categorical(y_dummy, num_classes=4)

history = model.fit(X_dummy, y_dummy, epochs=3)
img_input = np.expand_dims(img/255.0, axis=0)

prediction = model.predict(img_input)
stage = np.argmax(prediction)

stages = ["Stage 1", "Stage 2", "Stage 3", "Stage 4"]

print("Predicted Cancer Stage:", stages[stage])
confidence = np.max(prediction) * 100
print("Model Confidence (%):", confidence)

print("Estimated Cancer Affect Level (%):", size_percentage)
print("Training Accuracy:", history.history['accuracy'][-1] * 100)