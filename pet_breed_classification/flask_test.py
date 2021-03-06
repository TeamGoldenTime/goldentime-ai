import json
import requests
import numpy as np
from PIL import Image

image = Image.open('dog_pictures/dog_test4.jpeg')
pixels = np.array(image)

headers = {'Content-Type':'application/json'}
address = "http://127.0.0.1:5000/inference"
data = {'images':pixels.tolist()}

result = requests.post(address, data=json.dumps(data), headers=headers)

print(str(result.content, encoding='utf-8'))