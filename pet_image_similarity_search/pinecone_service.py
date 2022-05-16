import os
import torch
import pandas as pd
import numpy as np
import torchvision
import torch.nn as nn
from torch.utils.data import Dataset, random_split, DataLoader
import torch.nn.functional as F
from torchvision.utils import make_grid
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
from torchvision import transforms as ts
import torchvision.models as models
from PIL import Image
from collections import OrderedDict
import pinecone
import os

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
index_name = "simple-pytorch-dog-search"

# Checking whether the index already exists.
if index_name not in pinecone.list_indexes():
    pinecone.create_index(index_name, dimension=1000, metric="euclidean") # let's try cosine similiarty

index = pinecone.Index(index_name=index_name) # connect the index

# 일부를 인덱스에 upsert한다.
# for batch in chunks(zip(item_df.embedding_id, item_df.embedding), 50):
#     index.upsert(vectors=batch)

# new image
query_image_path = "pet_breed_classification/dog_pictures/cocker_spaniel.jpeg"

# embedder class
class ImageEmbedder:
    def __init__(self):
        self.normalize = ts.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
        # see https://pytorch.org/vision/0.8/models.html for many more model options
        self.model = models.squeezenet1_0(pretrained=True, progress=False)  # squeezenet

    def embed(self, image):
        # query_image = Image.open(query_image_path).convert("RGB")
        image = ts.Resize(256)(image)
        image = ts.CenterCrop(224)(image)
        tensor = ts.ToTensor()(image)
        tensor = self.normalize(tensor).reshape(1, 3, 224, 224)
        vector = self.model(tensor).cpu().detach().numpy().flatten()
        return vector

image_embedder = ImageEmbedder()

# extract features
query_image = Image.open(query_image_path).convert("RGB")
query_image_embedding = image_embedder.embed(query_image).tolist()

# search similarity image
df = pd.DataFrame()
df["embedding"] = [
    query_image_embedding
]
result = index.query(df.embedding, top_k = 5)
print(result)

# 결과 이미지 출력을 어떻게 해결하지? 디비에 있는 이미지의 아이디와 같게 해야겠다.
# res = index.query(batch, top_k=10)  # issuing queries #이 부분 어떤 식으로 보내줘야하지?
# total_res += [res.matches for res in res.results] # 이 부분으로 결과를 받아온다.


