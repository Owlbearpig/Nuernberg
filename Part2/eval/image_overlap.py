from constants import painting_photo_path
from PIL import Image
import numpy as np

background = Image.open('Intensity_0.5-0.8THz.png')#Image.open(painting_photo_path)
foreground = Image.open(painting_photo_path)#Image.open('Intensity_0.5-0.8THz.png')

Image.alpha_composite(background, foreground).save("test3.png")

