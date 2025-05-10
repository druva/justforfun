from __future__ import absolute_import, division, print_function
import os, subprocess
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import logging
logger = tf.get_logger()
logger.setLevel(logging.ERROR)


_URL = 'https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip'
zip_dir = tf.keras.utils.get_file('cats_and_dogs.zip', origin=_URL, extract=True)

zip_dir_base = os.path.dirname(zip_dir)
base_dir = os.path.join(zip_dir_base, 'cats_and_dogs_extracted/cats_and_dogs_filtered')

print("zip_dir_base", zip_dir_base)
print("base_dir", base_dir)