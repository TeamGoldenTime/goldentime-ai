import json
import requests
import numpy as np
from PIL import Image

image = Image.open("pet_ai/pet_test_pictures/Shih-Tzu.jpeg")
pixels = np.array(image)

headers = {'Content-Type':'application/json'}
similarity_address = "http://127.0.0.1:5000/image_similarity_inference"
classification_address = "http://127.0.0.1:5000/inference"
data = {'images':pixels.tolist()}

similarity_result = requests.post(similarity_address, data=json.dumps(data), headers=headers)
classification_result = requests.post(classification_address, data=json.dumps(data), headers=headers)

print(str(similarity_result.content, encoding='utf-8'))
print(str(classification_result.content, encoding='utf-8'))