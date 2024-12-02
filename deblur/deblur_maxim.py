from huggingface_hub import from_pretrained_keras
from PIL import Image

import tensorflow as tf
import numpy as np
import requests

import re


def deblur(img):
    if re.match(r'^https?://', img):
        image = Image.open(requests.get(img, stream=True).raw)
        image = np.array(image)
    else:
        image = np.array(img)

    image = tf.convert_to_tensor(image)
    image = tf.image.resize(image, (256, 256))
    model = from_pretrained_keras("google/maxim-s3-deblurring-reds")
    predictions = model.predict(tf.expand_dims(image, 0))
    return predictions

result = deblur("https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSBljkDcdonFKtzz7FRYqZ-_zZgC8DqdmS14g&s")
print(result)