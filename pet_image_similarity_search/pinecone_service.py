import pinecone
import os

api_key = os.getenv("PINECONE_API_KEY") or "4e5fd852-89e4-44fb-9e33-d605b79aba2f"
pinecone.init(api_key=api_key, environment='us-west1-gcp')

import pinecone.info

version_info = pinecone.info.version()
server_version = ".".join(version_info.server.split(".")[:2])
client_version = ".".join(version_info.client.split(".")[:2])

assert client_version == server_version, "Please upgrade pinecone-client."

# Choosing an arbitrary name for my index
index_name = "simple-pytorch-dog-search"

# Checking whether the index already exists.
if index_name not in pinecone.list_indexes():
    pinecone.create_index(index_name, dimension=1000, metric="euclidean")

index = pinecone.Index(index_name=index_name)


import itertools

def chunks(iterable, batch_size=100):
    it = iter(iterable)
    chunk = tuple(itertools.islice(it, batch_size))
    while chunk:
        yield chunk
        chunk = tuple(itertools.islice(it, batch_size))

# for batch in chunks(zip(item_df.embedding_id, item_df.embedding), 50):
#     index.upsert(vectors=batch)

import time


start = time.perf_counter()
total_res = list()
for batch in chunks(query_df.embedding,10):
    res = index.query(batch, top_k=10)  # issuing queries
    total_res += [res.matches for res in res.results]

end = time.perf_counter()
print("Run this test on a fast network to get the best performance.")

# from torchvision import transforms as ts
# import torchvision.models as models


# class ImageEmbedder:
#     def __init__(self):
#         self.normalize = ts.Normalize(
#             mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
#         )
#         # see https://pytorch.org/vision/0.8/models.html for many more model options
#         self.model = models.squeezenet1_0(pretrained=True, progress=False)  # squeezenet

#     def embed(self, image_file_name):
#         image = Image.open(image_file_name).convert("RGB")
#         image = ts.Resize(256)(image)
#         image = ts.CenterCrop(224)(image)
#         tensor = ts.ToTensor()(image)
#         tensor = self.normalize(tensor).reshape(1, 3, 224, 224)
#         vector = self.model(tensor).cpu().detach().numpy().flatten()
#         return vector


# image_embedder = ImageEmbedder()

