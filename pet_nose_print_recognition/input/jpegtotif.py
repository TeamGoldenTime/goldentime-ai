from PIL import Image
import glob
import os

for image in glob.glob("pet_nose_print_recognition/input/*.jpeg"):
    im = Image.open(image)
    file_name = os.path.split(image)[-1].split('.')[0]
    file = os.path.split(image)[0] + '/' + file_name + '.tiff'
    im.save(file, 'TIFF')