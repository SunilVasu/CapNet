import os
import cv2
import numpy as np

PATH = "./gan_dataset_resized/"
images = []
labels = []
for dir, subdir, files in os.walk(PATH):
    for file in files:
        if file.endswith(".jpg"):
            dir_parts = dir.split("/")
            label = int(dir_parts[-1])
            img = cv2.imread(os.path.join(dir, file), 0)
            img = img.flatten()
            labels.append(label)
            images.append(img)
images = np.stack(images)
labels = np.array(labels)

print images.shape
print images
print labels

nb_classes = 10
targets = labels.reshape(-1)
one_hot_targets = np.eye(nb_classes)[targets]

print one_hot_targets

np.save("labeled_sample2.npy", {"images": images, "labels": one_hot_targets})
