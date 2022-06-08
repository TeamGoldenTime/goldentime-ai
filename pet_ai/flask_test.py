import json
import requests
import numpy as np
from PIL import Image

from urllib import request
from io import BytesIO


# url 이미지 (종분류는 jpg만 가능) / 말티즈 푸들 위주로
# 방금 저장한 이미지
# path = "https://www.animal.go.kr/front/fileMng/imageView.do?f=/files/shelter/2022/04/202206071806272[1].jpg"

# 방금 저장한 이미지를 검색할 수 있을까? ㅇㅇ
path = "https://www.animal.go.kr/front/fileMng/imageView.do;jsessionid=IuI5eVLaVeAChW9Tt61nfU9kAqKAadrwihATepAr0Du7jdzP7i2FHj7vqJpPbv6p.aniwas2_servlet_front?f=/files/shelter/2022/04/202205170905760.png"

# res = request.urlopen(path).read()
# image = Image.open(BytesIO(res))

# 로컬 이미지
# image = Image.open("pet_ai/pet_test_pictures/Shih-Tzu.jpeg")

headers = {'Content-Type':'application/json'}
similarity_address = "http://127.0.0.1:5000/image_similarity_inference" #image_similarity_update
similarity_update_address = "http://127.0.0.1:5000/image_similarity_update"
classification_address = "http://127.0.0.1:5000/inference"

# pixels = np.array(image)
data = {'path': path}

# 아래가 업데이트해주는 메소드
# _ = requests.put(similarity_update_address, data=json.dumps(data), headers=headers)
similarity_result = requests.post(similarity_address, data=json.dumps(data), headers=headers)
classification_result = requests.post(classification_address, data=json.dumps(data), headers=headers)

print(str(similarity_result.content, encoding='utf-8'))
print(str(classification_result.content, encoding='utf-8'))