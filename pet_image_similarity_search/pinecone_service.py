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

# Choosing an arbitrary name for my index: 인덱스 이름 지정
index_name = "simple-pytorch-dog-search"

# Checking whether the index already exists.: 인덱스 존재여부확인
if index_name not in pinecone.list_indexes():
    pinecone.create_index(index_name, dimension=1000, metric="euclidean") #없으면 만들기 -> 여기서 cosine similarity 사용 가능

index = pinecone.Index(index_name=index_name) # 인덱스 연결하기

# 일부를 인덱스에 upsert한다.
# for batch in chunks(zip(item_df.embedding_id, item_df.embedding), 50):
#     index.upsert(vectors=batch)

# 요렇게 해주면 된다.
index.query(query_df[:1].embedding, top_k = 2)
    # res = index.query(batch, top_k=10)  # issuing queries #이 부분 어떤 식으로 보내줘야하지?
    # total_res += [res.matches for res in res.results] # 이 부분으로 결과를 받아온다.


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

