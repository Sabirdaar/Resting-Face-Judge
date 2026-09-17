from pathlib import Path

import numpy as np
from PIL import Image


DATASET_PATH = Path("data/raw/fer2013")

TRAIN_PATH = DATASET_PATH / "train"
TEST_PATH = DATASET_PATH / "test"


# --------------------------------------------------
# Find the classes
# --------------------------------------------------

classes = sorted(
    folder.name
    for folder in TRAIN_PATH.iterdir()
    if folder.is_dir()
)

class_to_number = {
    class_name: number
    for number, class_name in enumerate(classes)
}

print("Classes:")
print(class_to_number)


# --------------------------------------------------
# Load training data
# --------------------------------------------------

X_train = []
y_train = []

for class_name in classes:
    class_path = TRAIN_PATH / class_name

    for image_path in class_path.glob("*.jpg"):
        image = Image.open(image_path)
        pixels = np.array(image)

        X_train.append(pixels)
        y_train.append(class_to_number[class_name])


X_train = np.array(X_train)
y_train = np.array(y_train)


# --------------------------------------------------
# Load test data
# --------------------------------------------------

X_test = []
y_test = []

for class_name in classes:
    class_path = TEST_PATH / class_name

    for image_path in class_path.glob("*.jpg"):
        image = Image.open(image_path)
        pixels = np.array(image)

        X_test.append(pixels)
        y_test.append(class_to_number[class_name])


X_test = np.array(X_test)
y_test = np.array(y_test)


# --------------------------------------------------
# Dataset information
# --------------------------------------------------

print("\nTraining dataset:")
print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)

print("\nTest dataset:")
print("X_test shape:", X_test.shape)
print("y_test shape:", y_test.shape)


print("\nData types:")
print("X_train dtype:", X_train.dtype)
print("y_train dtype:", y_train.dtype)
print("X_test dtype:", X_test.dtype)
print("y_test dtype:", y_test.dtype)


print("\nPixel range:")
print("Training min:", X_train.min())
print("Training max:", X_train.max())
print("Test min:", X_test.min())
print("Test max:", X_test.max())