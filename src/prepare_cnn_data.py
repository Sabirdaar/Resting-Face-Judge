import os
from pathlib import Path
import numpy as np
from PIL import Image

try:
    os.environ.setdefault("KERAS_BACKEND", "torch")
    from keras.utils import to_categorical
except ImportError:
    try:
        from tensorflow.keras.utils import to_categorical
    except ImportError:

        def to_categorical(y, num_classes=7):
            return np.eye(num_classes, dtype="float32")[y]


DATASET_PATH = Path("data/raw/fer2013")
TRAIN_PATH = DATASET_PATH / "train"
TEST_PATH = DATASET_PATH / "test"
PROCESSED_PATH = Path("data/processed")


def main():
    classes = sorted(
        folder.name for folder in TRAIN_PATH.iterdir() if folder.is_dir()
    )
    class_to_number = {
        class_name: number for number, class_name in enumerate(classes)
    }

    print("Classes:")
    print(class_to_number)

    # Load raw training data
    print("\nLoading training images...")
    X_train = []
    y_train = []
    for class_name in classes:
        class_path = TRAIN_PATH / class_name
        for image_path in class_path.glob("*.jpg"):
            image = Image.open(image_path)
            X_train.append(np.array(image))
            y_train.append(class_to_number[class_name])

    X_train = np.array(X_train)
    y_train = np.array(y_train)

    # Load raw test data
    print("Loading test images...")
    X_test = []
    y_test = []
    for class_name in classes:
        class_path = TEST_PATH / class_name
        for image_path in class_path.glob("*.jpg"):
            image = Image.open(image_path)
            X_test.append(np.array(image))
            y_test.append(class_to_number[class_name])

    X_test = np.array(X_test)
    y_test = np.array(y_test)

    # --------------------------------------------------
    # User Request: Reshape & Normalize for CNN + One-Hot Encoding
    # --------------------------------------------------
    y_train_onehot = to_categorical(y_train, num_classes=7)
    y_test_onehot = to_categorical(y_test, num_classes=7)

    X_train_cnn = X_train.reshape(*X_train.shape, 1).astype("float32") / 255.0
    X_test_cnn = X_test.reshape(*X_test.shape, 1).astype("float32") / 255.0

    print("\n================ CNN Dataset Ready ================")
    print(f"X_train_cnn shape:   {X_train_cnn.shape} | dtype: {X_train_cnn.dtype}")
    print(f"y_train_onehot shape:{y_train_onehot.shape} | dtype: {y_train_onehot.dtype}")
    print(f"X_test_cnn shape:    {X_test_cnn.shape} | dtype: {X_test_cnn.dtype}")
    print(f"y_test_onehot shape: {y_test_onehot.shape} | dtype: {y_test_onehot.dtype}")
    print(f"Pixel Range:         [{X_train_cnn.min():.1f}, {X_train_cnn.max():.1f}]")
    print("===================================================")

    PROCESSED_PATH.mkdir(parents=True, exist_ok=True)
    save_path = PROCESSED_PATH / "fer2013_cnn.npz"
    np.savez_compressed(
        save_path,
        X_train_cnn=X_train_cnn,
        y_train_onehot=y_train_onehot,
        X_test_cnn=X_test_cnn,
        y_test_onehot=y_test_onehot,
    )
    print(f"\nSaved preprocessed CNN dataset to: {save_path.resolve()}")


if __name__ == "__main__":
    main()
