import numpy as np
import math

class ImageFilters:
    @staticmethod
    def gaussian_kernel(size, sigma=1.0):
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
