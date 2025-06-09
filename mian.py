from images import ImageLoader
from image_filters import ImageFilters
from PIL import Image

# Wczytaj obraz
loader = ImageLoader("obraz.bmp")
img = loader.load_grayscale()

# Filtry
blurred = ImageFilters.gaussian_blur(img, size=5, sigma=1.0)
medianed = ImageFilters.median_filter(img, size=3)
equalized = ImageFilters.histogram_equalization(img)

# Zapis wyników
Image.fromarray(blurred).save("gauss.png")
Image.fromarray(medianed).save("median.png")
Image.fromarray(equalized).save("equalized.png")