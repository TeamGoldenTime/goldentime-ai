import json
import requests
import numpy as np
from PIL import Image

image = Image.open('pet_breed_classification/dog_pictures/dog_test4.jpeg')
pixels = np.array(image)

headers = {'Content-Type':'application/json'}
address = "http://127.0.0.1:5000/image_similarity_inference"
data = {'images':pixels.tolist()}

result = requests.post(address, data=json.dumps(data), headers=headers)

print(str(result.content, encoding='utf-8'))