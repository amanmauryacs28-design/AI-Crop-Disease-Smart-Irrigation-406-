
from pathlib import Path
import json
import numpy as np
import tensorflow as tf
from tensorflow import keras

BASE_DIR = Path(__file__).resolve().parent
TEST_DIR = BASE_DIR / "test"
MODEL_PATH = BASE_DIR / "models" / "crop_disease_model.keras"
CLASS_PATH = BASE_DIR / "models" / "class_names.json"

IMAGE_SIZE = (128, 128)
BATCH_SIZE = 8

# Check required files
if not TEST_DIR.exists():
    raise FileNotFoundError(f"Test folder not found: {TEST_DIR}")

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

# Load saved model and training class names
print("Loading saved model...")
model = keras.models.load_model(MODEL_PATH)

with open(CLASS_PATH, "r") as f:
    saved_classes = json.load(f)

# Load test images without shuffling
print("Loading test dataset...")
test_ds = keras.utils.image_dataset_from_directory(
    TEST_DIR,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int",
    shuffle=False
)

test_classes = test_ds.class_names

if saved_classes != test_classes:
    raise ValueError(
        f"Class order mismatch!\n"
        f"Saved: {saved_classes}\n"
        f"Test: {test_classes}"
    )

print("\nClasses:", saved_classes)
print("Evaluating model on test images...")

# Overall test loss and accuracy
loss, accuracy = model.evaluate(test_ds, verbose=1)

print(f"\nTest Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy * 100:.2f}%")

# Get predictions for per-class evaluation
all_labels = np.concatenate([
    labels.numpy() for _, labels in test_ds
])

probabilities = model.predict(test_ds, verbose=1)
predicted_labels = np.argmax(probabilities, axis=1)

correct = predicted_labels == all_labels

print("\nPer-class results:")
for class_index, class_name in enumerate(saved_classes):
    mask = all_labels == class_index
    total = int(np.sum(mask))
    class_correct = int(np.sum(correct & mask))

    class_accuracy = (
        class_correct / total * 100 if total else 0
    )

    print(
        f"{class_name}: "
        f"{class_correct}/{total} correct "
        f"({class_accuracy:.2f}%)"
    )

print("\nEvaluation completed.")
