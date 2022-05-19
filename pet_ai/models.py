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

breeds = ['치와와',
 '시츄', #Japanese spaniel
 '말티즈',
 '페키니즈',
 '시츄',
 '킹 찰스 스패니얼',
 '파피용',
 '토이 테리어',
 '로다지안 리지백',
 '아프간 하운드',
 '바셋 하운드',
 '비글',
 '블러드 하운드',
 '블루틱 쿤하운드',
 '블랙 앤 탄 쿤하운드',
 '워커 쿨하운드',
 '잉글리시 폭스하운드',
 '레드본 쿤하운드',
 '보르조이',
 '아이리시 울프하운드',
 '이탈리안 그레이하운드',
 '휘핏',
 '이비전 하운드',
 '노르웨이언 엘크하운드',
 '오터 하운드',
 '살루키',
 '스코티시 디어하운드',
 '와이머라너',
 '스타포드셔 불 테리어',
 '아메리칸 스태퍼드셔 테리어',
 '베들링턴 테리어',
 '보더 테리어',
 '케리 블루 테리어',
 '아이리시 테리어',
 '노퍽 테리어',
 '노리치 테리어',
 '요크셔 테리어',
 '와이어 폭스 테리어',
 '레이클랜드 테리어',
 '실리엄 테리어',
 '에어데일 테리어',
 '케언 테리어',
 '오스트레일리안 테리어',
 '댄디 딘몬트 테리어',
 '보스턴 테리어',
 '미니어처 슈나우저',
 '자이언트 슈나우저',
 '스탠더드 슈나우저',
 '스코티시 테리어',
 '티베탄 테리어',
 '오스트레일리안 실키 테리어',
 '아이리쉬 소프트코티드 휘튼 테리어',
 '웨스트 하일랜드 화이트 테리어',
 '라사압소',
 '플랫 코티드 리트리버',
 '컬리 코티드 리트리버',
 '골든리트리버', # 골든리트리버
 '래브라도 리트리버',
 '체서피크 베이 리트리버',
 '저먼 쇼트헤어드 포인터',
 '비즐라',
 '르웰린',
 '아이리시 세터',
 '고든 세터',
 '브리트니',
 '클럼버 스파니엘',
 '잉글리시 스프링어 스패니얼',
 '웰시 스프링어 스패니얼',
 '잉글리시 코커 스패니얼',
 '서식스 스패니얼',
 '아이리시 워터 스패니얼',
 '쿠바츠',
 '스키퍼키',
 '그루넨달',
 '말리노이즈',
 '브리아드',
 '켈피',
 '코몬돌',
 '올드 잉글리시 쉽독',
 '셔틀랜드 쉽독',
 '콜리',
 '보더 콜리',
 '부비에 데 플랑드르',
 '로트바일러',
 '저먼 셰퍼드',
 '도베르만 핀셔',
 '미니어처 핀셔',
 '그레이터 스위스 마운틴 도그',
 '버니즈 마운틴 도그',
 '아펜젤러 세넨훈드',
 '엔틀버쳐 마운틴 독',
 '복서',
 '불마스티프',
 '티베탄 마스티프',
 '프렌치 불도그',
 '그레이트 데인',
 '세인트버나드',
 '에스키모 도그',
 '맬러뮤트',
 '시베리안 허스키',
 '아펜핀셔',
 '바센지',
 '퍼그',
 '레온베르거',
 '뉴펀들랜드',
 '그레이트 피레니즈',
 '사모예드견',
 '포메라니안',
 '차우 차우',
 '키스혼드',
 '브뤼셀 그리펀',
 '펨브로크',
 '카디컨',
 '토이푸들',
 '미니어처 푸들',
 '스탠다드 푸들',
 '멕시칸 헤어리스 도그',
 '딩고',
 '돌',
 '아프리카 들개']

def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))

class ImageClassificationBase(nn.Module):
    # training step
    def training_step(self, batch):
        img, targets = batch
        out = self(img)
        loss = F.nll_loss(out, targets)
        return loss
    
    # validation step
    def validation_step(self, batch):
        img, targets = batch
        out = self(img)
        loss = F.nll_loss(out, targets)
        acc = accuracy(out, targets)
        return {'val_acc':acc.detach(), 'val_loss':loss.detach()}
    
    # validation epoch end
    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()
        return {'val_loss':epoch_loss.item(), 'val_acc':epoch_acc.item()}
        
    # print result end epoch
    def epoch_end(self, epoch, result):
        print("Epoch [{}] : train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(epoch, result["train_loss"], result["val_loss"], result["val_acc"]))

class DogBreedPretrainedWideResnet(ImageClassificationBase):
    def __init__(self):
        super().__init__()
        
        self.network = models.wide_resnet50_2(pretrained=True)
        # Replace last layer
        num_ftrs = self.network.fc.in_features
        self.network.fc = nn.Sequential(
            nn.Linear(num_ftrs, 120),
            nn.LogSoftmax(dim=1)
        )
        
    def forward(self, xb):
        return self.network(xb)

def get_default_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')

def to_device(data, device):
    if isinstance(data, (list, tuple)):
        return [to_device(d, device) for d in data]
    else:
        return data.to(device, non_blocking=True)

class DeviceDataLoader:
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device
    
    def __len__(self):
        return len(self.dl)
    
    def __iter__(self):
        for batch in self.dl:
            yield to_device(batch, self.device)

def predict_new(img, model, device):
    test_transform = transforms.Compose([
      transforms.Resize((224,224)), 
      transforms.ToTensor(),
    #    transforms.Normalize(*imagenet_stats, inplace=True)
    ])

    img = test_transform(img)
    xb = img.unsqueeze(0)
    xb = to_device(xb, device)
    preds = model(xb)
    predictions = preds[0]
    max_val, kls = torch.topk(predictions, k=3, dim= 0) #torch.max(predictions, dim=0) 
    print('Predicted :', breeds[kls[0]],',',breeds[kls[1]],',',breeds[kls[2]])
    # plt.imshow(img.permute(1,2,0))
    # plt.show()
    return breeds[kls[0]], breeds[kls[1]], breeds[kls[2]]


# embedder class
class ImageEmbedder:
    def __init__(self):
        self.normalize = ts.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
        # see https://pytorch.org/vision/0.8/models.html for many more model options
        self.model = models.squeezenet1_0(pretrained=True, progress=False)  # squeezenet

    def embed(self, image):
        # image = Image.open(image_file_name).convert("RGB")
        image = ts.Resize(256)(image)
        image = ts.CenterCrop(224)(image)
        tensor = ts.ToTensor()(image)
        tensor = self.normalize(tensor).reshape(1, 3, 224, 224)
        vector = self.model(tensor).cpu().detach().numpy().flatten()
        return vector