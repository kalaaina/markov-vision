# ------------------------------------------------------------
# image_loader.py
#
# This module is responsible for loading and saving images
# for the Markov-Vision project.
#
# Functions:
# - load_image(path):
#     Opens an image, converts it to RGB, resizes it to
#     128x128 pixels, and returns it as a NumPy array.
#
# - save_image(arr, path):
#     Converts a NumPy array back into an image and saves
#     it to the specified path.
#
# This file is used by the main application to read input
# images and store the processed results.
# ------------------------------------------------------------
from PIL import Image
import numpy as np

IMAGE_SIZE = (128, 128)


def load_image(path):
    image = Image.open(path)
    image = image.convert("RGB")
    image = image.resize(IMAGE_SIZE)
    return np.array(image)


def save_image(arr, path):
    image = Image.fromarray(arr.astype(np.uint8))
    image.save(path)
