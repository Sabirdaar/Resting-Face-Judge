from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)


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
# Flatten and normalize data
# --------------------------------------------------

X_train_flat = X_train.reshape(X_train.shape[0], -1) 
X_test_flat = X_test.reshape(X_test.shape[0], -1)


# --------------------------------------------------
# Preprocessed dataset information
# --------------------------------------------------

print("\nFlattened & normalized training dataset:")
print("X_train_flat shape:", X_train_flat.shape)
print("y_train shape:     ", y_train.shape)

print("\nFlattened & normalized test dataset:")
print("X_test_flat shape: ", X_test_flat.shape)
print("y_test shape:      ", y_test.shape)

print("\nData types:")
print("X_train_flat dtype:", X_train_flat.dtype)
print("X_test_flat dtype: ", X_test_flat.dtype)

print("\nNormalized pixel range:")
print("Training min:", X_train_flat.min())
print("Training max:", X_train_flat.max())
print("Test min:    ", X_test_flat.min())
print("Test max:    ", X_test_flat.max())


# --------------------------------------------------
# Random Forest Classifier Model
# --------------------------------------------------

class_names = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

print("\nTraining Random Forest Classifier...")
clf_rf = RandomForestClassifier(
    n_estimators=100,
    max_depth=15,
    class_weight='balanced',
    n_jobs=-1,
    random_state=42,
)
clf_rf.fit(X_train_flat, y_train)

train_acc_rf = clf_rf.score(X_train_flat, y_train)
test_acc_rf = clf_rf.score(X_test_flat, y_test)
print(f"\nTrain accuracy: {train_acc_rf:.4f}")
print(f"Test accuracy:  {test_acc_rf:.4f}\n")

y_pred_rf = clf_rf.predict(X_test_flat)
print(classification_report(y_test, y_pred_rf, target_names=class_names))

cm_rf = confusion_matrix(y_test, y_pred_rf)
disp = ConfusionMatrixDisplay(confusion_matrix=cm_rf, display_labels=class_names)
disp.plot(xticks_rotation=45, cmap='Blues')
plt.title("Random Forest — Confusion Matrix")
plt.tight_layout()
plt.show()



