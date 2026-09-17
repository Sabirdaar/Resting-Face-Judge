from pathlib import Path

import numpy as np
from PIL import Image


DATASET_PATH = Path("data/raw/fer2013/train")

image_path = next((DATASET_PATH / "happy").glob("*.jpg"))

image = Image.open(image_path)

pixels = np.array(image)


print("Image:")
print(image)

print("\nNumPy array:")
print(pixels)

print("\nShape:")
print(pixels.shape)

print("\nData type:")
print(pixels.dtype)

print("\nMinimum pixel value:")
print(pixels.min())

print("\nMaximum pixel value:")
print(pixels.max())

print("\nMean pixel value:")
print(pixels.mean())