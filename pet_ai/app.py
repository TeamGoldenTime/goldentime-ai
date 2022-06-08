from flask import request, Flask, jsonify

import os
import torch
import pandas as pd
import numpy as np
import torchvision
import torch.nn as nn
from torch.utils.data import Dataset, random_split, DataLoader
import torch.nn.functional as F
from torchvision.utils import make_grid
import torchvision.transforms as transforms
from torchvision import transforms as ts
import torchvision.models as models
from PIL import Image
from collections import OrderedDict
import pinecone
from urllib import request as url_request
from io import BytesIO

import models


app = Flask(__name__)

#1. pet_classification
model = models.DogBreedPretrainedWideResnet()
weights_fname = 'models/dog-breed-classifier-wideresnet_with_data_aug.pth'
model.load_state_dict(torch.load(weights_fname, map_location=torch.device('cpu')))
model.eval()

device = models.get_default_device()
models.to_device(model, device)


#2. pet_search
# api_key
api_key = os.getenv("PINECONE_API_KEY") or "4e5fd852-89e4-44fb-9e33-d605b79aba2f"
pinecone.init(api_key=api_key, environment='us-west1-gcp')

# version compatability
import pinecone.info
version_info = pinecone.info.version()
server_version = ".".join(version_info.server.split(".")[:2])
client_version = ".".join(version_info.client.split(".")[:2])

assert client_version == server_version, "Please upgrade pinecone-client."

# Choosing an arbitrary name for my index
# index_name = "simple-pytorch-dog-search"
index_name = "simple-pytorch-image-search"

# Checking whether the index already exists.
if index_name not in pinecone.list_indexes():
    pinecone.create_index(index_name, dimension=1000, metric="euclidean") # let's try cosine similiarty

index = pinecone.Index(index_name=index_name) # connect the index
image_embedder = models.ImageEmbedder()


#3. function
@app.route('/inference', methods=['POST'])
def inference():
    data = request.json
    path = data['path']
    res = url_request.urlopen(path).read()
    img = Image.open(BytesIO(res))
    # numpy_data = np.array(data['images'], dtype=np.uint8)
    # img = Image.fromarray(numpy_data)
    result = models.predict_new(img, model, device)
    return jsonify(breed = result[0])


# 성능이 안좋아지는게 아니고 백엔드에 없는 이미지를 인풋으로 넣어줘서 성능이 떨어진거였다 아놔 --..
@app.route('/image_similarity_inference', methods=['POST'])
def image_similarity_inference():
    query_data = request.json
    path = query_data['path']
    # query_numpy_data = np.array(query_data['images'], dtype=np.uint8)
    # query_image = Image.fromarray(query_numpy_data).convert("RGB")
    res = url_request.urlopen(path).read()
    query_image = Image.open(BytesIO(res)).convert("RGB")
    # extract features
    query_image_embedding = image_embedder.embed(query_image).tolist()

    # search similarity image
    df = pd.DataFrame()
    df["embedding"] = [
        query_image_embedding
    ]
    result = index.query(df.embedding, top_k = 10)
    data = []
    for obj in result.results[0].matches :
        data.append({"id" : obj.id, "score" : obj.score});
        
    print(data)
    return jsonify(data)


    # path = "https://www.animal.go.kr/front/fileMng/imageView.do;jsessionid=Cx3VGauMMb8y3UTX38E1XLjeXVVePpvT1nF7jeQLiaVNcCtk7B5gpiUAGFfOaX1S.aniwas2_servlet_front?f=/files/loss/2022/05/20220510102404490_s.jpg"
    # res = request.urlopen(path).read()
    # img = Image.open(BytesIO(res))

@app.route('/image_similarity_update', methods=['PUT'])
def image_similarity_update():
    query_data = request.json
    path = query_data['path']
    # query_numpy_data = np.array(query_data['images'], dtype=np.uint8)
    # query_image = Image.fromarray(query_numpy_data).convert("RGB")
    res = url_request.urlopen(path).read()
    query_image = Image.open(BytesIO(res)).convert("RGB")
    # extract features
    query_image_embedding = image_embedder.embed(query_image).tolist()

    import itertools
    def chunks(iterable, batch_size=1):
        it = iter(iterable)
        chunk = tuple(itertools.islice(it, batch_size))
        while chunk:
            yield chunk
            chunk = tuple(itertools.islice(it, batch_size))

    # # search similarity image
    df = pd.DataFrame()
    df["embedding"] = [
        query_image_embedding
    ]
    df["embedding_id"] = [
        path
    ]
    # index.upsert(id = path, vectors = query_image_embedding)

    for batch in chunks(zip(df.embedding_id, df.embedding), 1):
        index.upsert(vectors=batch)

    return jsonify({"message":"update data successfully"})
    
if __name__ == '__main__':
    app.run(debug=True)