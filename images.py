import numpy as np
from PIL import Image



class ImageLoader:
    def __init__(self, path):
        self.path = path
        self.image_array = None

    def load_grayscale(self):
        image = Image.open(self.path).convert("L")
        self.image_array = np.array(image, dtype=np.uint8)
        return self.image_array
