from transformers import pipeline
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from functools import lru_cache

@lru_cache
def get_depth(image_path):
    # modelname = "depth-anything/Depth-Anything-V2-Small-hf"
    # modelname = "depth-anything/Depth-Anything-V2-Base-hf"
    modelname = "depth-anything/Depth-Anything-V2-Large-hf"


    pipe = pipeline(task="depth-estimation", model=modelname)
    image = Image.open(image_path)
    depth = pipe(image)["depth"]    # ---> PIL.Image.Image
    depth = np.asarray(depth)       # ---> numpy.ndarray, value range = [0, 255]

    # plt.imshow(depth)
    # plt.show()
    # plt.savefig(image_path.split('.')[-2] + '_' + modelname.split('/')[1] +'.jpg', bbox_inches='tight')
    return depth

# get_depth('images/nguyen.png')

