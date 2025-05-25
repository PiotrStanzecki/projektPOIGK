import numpy as np
import math

class ImageFilters:
    @staticmethod
    def gaussian_kernel(size, sigma=0.5):
        kernel = np.zeros((size, size), dtype=np.float32)
        center = size // 2
        for i in range(size):
            for j in range(size):
                x = i - center
                y = j - center
                kernel[i, j] = math.exp(-(x**2 + y**2) / (2 * sigma**2))
        return kernel / np.sum(kernel)

    @staticmethod
    def apply_kernel(image, kernel):
        h, w = image.shape
        k = kernel.shape[0] // 2
        output = np.zeros_like(image, dtype=np.uint8)
        for i in range(k, h - k):
            for j in range(k, w - k):
                region = image[i - k:i + k + 1, j - k:j + k + 1]
                value = np.sum(region * kernel)
                output[i, j] = np.clip(value, 0, 255)
        return output

    @staticmethod
    def gaussian_blur(image, size=5, sigma=1.0):
        kernel = ImageFilters.gaussian_kernel(size, sigma)
        return ImageFilters.apply_kernel(image, kernel)
