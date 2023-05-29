import numpy as np
from PIL import Image


Image_name = "a.png"


img = Image.open(Image_name)


img_array = np.array(img)
gray_values = img_array.ravel()


min_value = np.min(gray_values)
max_value = np.max(gray_values)
range_value = max_value - min_value

#histogram sretching
stretched_array = ((img_array - min_value) / range_value) * 255

img_stretched = Image.fromarray(stretched_array.astype(np.uint8))

img_stretched.save(f"final-{Image_name}")
