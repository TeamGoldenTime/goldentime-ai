import json
import requests
import numpy as np
from PIL import Image

from urllib import request
from io import BytesIO


# url 이미지 (종분류는 jpg만 가능) / 말티즈 푸들 위주로
path = "https://www.animal.go.kr/front/fileMng/imageView.do;jsessionid=VTdwc2Guq27aBaGVPDeDboFG7GNCoMMyD2QU8nSeM44rUBWO7QoWV5DWBwW8JKth.aniwas_servlet_front?f=/files/shelter/2022/04/202205160905846.jpeg"
# res = request.urlopen(path).read()
# image = Image.open(BytesIO(res))

# 로컬 이미지
# image = Image.open("pet_ai/pet_test_pictures/Shih-Tzu.jpeg")

headers = {'Content-Type':'application/json'}
similarity_address = "http://127.0.0.1:5000/image_similarity_inference"
classification_address = "http://127.0.0.1:5000/inference"
# pixels = np.array(image)
data = {'path': path}

similarity_result = requests.post(similarity_address, data=json.dumps(data), headers=headers)
classification_result = requests.post(classification_address, data=json.dumps(data), headers=headers)

print(str(similarity_result.content, encoding='utf-8'))
print(str(classification_result.content, encoding='utf-8'))