import numpy as np
import math
from scipy.ndimage import convolve

class ImageFilters:
  

    @staticmethod
    def gaussian_kernel(size, sigma=1.0):
 
        if size % 2 == 0:
            raise ValueError("Kernel size must be an odd number.")

        center = size // 2

        x = np.arange(-center, center + 1)
        y = np.arange(-center, center + 1)

        X, Y = np.meshgrid(x, y)

        kernel = np.exp(-(X**2 + Y**2) / (2 * sigma**2))

        return kernel / np.sum(kernel)

    @staticmethod
    def apply_kernel(image, kernel):
        convolved_image = convolve(image.astype(np.float32), kernel, mode='constant', cval=0.0)
        return np.clip(convolved_image, 0, 255).astype(np.uint8)


    @staticmethod
    def gaussian_blur(image, size=5, sigma=1.0):
        kernel = ImageFilters.gaussian_kernel(size, sigma)
        return ImageFilters.apply_kernel(image, kernel)

    @staticmethod
    def edge_detection(image, force):
        kernel_edge = np.array([[0,  force, 0], [force, -4 * force, force], [0,  force, 0]], dtype=np.float32)
        return ImageFilters.apply_kernel(image, kernel_edge)

    @staticmethod
    def median_filter(image, size=3):
        h, w = image.shape
        k = size // 2
        output = np.zeros_like(image, dtype=np.uint8)
        for i in range(k, h - k):
            for j in range(k, w - k):
                region = image[i - k:i + k + 1, j - k:j + k + 1]
                output[i, j] = np.median(region)
        return output

    @staticmethod
    def histogram_equalization(image):
        hist, _ = np.histogram(image.flatten(), bins=256, range=(0, 256))
        cdf = hist.cumsum()
        cdf_masked = np.ma.masked_equal(cdf, 0)
        cdf_normalized = (cdf_masked - cdf_masked.min()) * 255 / (cdf_masked.max() - cdf_masked.min())
        cdf_filled = np.ma.filled(cdf_normalized, 0).astype(np.uint8)
        return cdf_filled[image]

