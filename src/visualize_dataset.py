from pathlib import Path
import random

import matplotlib.pyplot as plt
from PIL import Image


DATASET_PATH = Path("data/raw/fer2013/train")

classes = sorted(
    folder.name
    for folder in DATASET_PATH.iterdir()
    if folder.is_dir()
)


fig, axes = plt.subplots(2, 4, figsize=(10, 5))

for ax, class_name in zip(axes.flat, classes):

    class_path = DATASET_PATH / class_name

    image_files = list(class_path.glob("*.jpg"))

    image_path = random.choice(image_files)

    image = Image.open(image_path)

    ax.imshow(image, cmap="gray")
    ax.set_title(class_name)
    ax.axis("off")


# Hide the unused 8th subplot
axes.flat[-1].axis("off")

plt.tight_layout()
plt.show()