import os
import json
import matplotlib.pyplot as plt

HISTORY_PATH = os.path.join("models", "history.json")
OUTPUT_DIR = "outputs"
ACCURACY_PATH = os.path.join(OUTPUT_DIR, "accuracy.png")
LOSS_PATH = os.path.join(OUTPUT_DIR, "loss.png")

os.makedirs(OUTPUT_DIR, exist_ok=True)

if not os.path.exists(HISTORY_PATH):
    raise FileNotFoundError(f"History file not found: {HISTORY_PATH}")

with open(HISTORY_PATH, "r") as f:
    history = json.load(f)

# Accuracy plot
plt.figure(figsize=(8, 5))
plt.plot(history["accuracy"], label="Train Accuracy")
plt.plot(history["val_accuracy"], label="Validation Accuracy")
plt.title("Model Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(ACCURACY_PATH)
plt.show()

# Loss plot
plt.figure(figsize=(8, 5))
plt.plot(history["loss"], label="Train Loss")
plt.plot(history["val_loss"], label="Validation Loss")
plt.title("Model Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(LOSS_PATH)
plt.show()

print(f"Accuracy graph saved at: {ACCURACY_PATH}")
print(f"Loss graph saved at: {LOSS_PATH}")