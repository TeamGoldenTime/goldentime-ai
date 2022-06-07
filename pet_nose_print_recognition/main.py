import preprocessing
import glob
from PIL import Image
import os
from tqdm import tqdm 


# 배경없앤다.
# for file in glob.glob("pet_nose_print_recognition/input/*.jpeg"):
#     preprocessing.remove_back(file)
#     # preprocessing.fingerprint_pipline(img, file)
#     print(f"{file}의 전처리가 끝났습니다")

# jpeg -> tiff
# for image in glob.glob("pet_nose_print_recognition/background/*.jpeg"):
#     im = Image.open(image)
#     file_name = os.path.split(image)[-1].split('.')[0]
#     file = os.path.split(image)[0] + '/' + file_name + '.tif'
#     im.save(file)

# preprocessing
img_dir = 'pet_nose_print_recognition/background/'
output_dir = 'pet_nose_print_recognition/output/'
images, images_paths = preprocessing.open_images(img_dir)
os.makedirs(output_dir, exist_ok=True)
for i, img in enumerate(tqdm(images)):
    results = preprocessing.fingerprint_pipline(img, images_paths[i])
    # cv.imwrite(output_dir+str(i)+'.png', results)

#이후에 pet_nose_matching 사용하기