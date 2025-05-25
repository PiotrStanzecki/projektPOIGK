from images import ImageLoader
from image_filters import ImageFilters
from PIL import Image

# Wczytanie obrazu
loader = ImageLoader("obraz.bmp")
img = loader.load_grayscale()

# Filtry
blurred = ImageFilters.gaussian_blur(img, size=5, sigma=10)


# Zapis wyników
Image.fromarray(blurred).save("gauss.png")