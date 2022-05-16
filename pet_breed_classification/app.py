
from flask import request, Flask, jsonify

import torch
import pandas as pd
import numpy as np
import torchvision
import torch.nn as nn
from torch.utils.data import Dataset, random_split, DataLoader
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torchvision.utils import make_grid
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
from collections import OrderedDict
# %matplotlib inline

breeds = ['Chihuahua',
 'Japanese spaniel',
 'Maltese dog',
 'Pekinese',
 'Shih Tzu',
 'Blenheim spaniel',
 'papillon',
 'toy terrier',
 'Rhodesian ridgeback',
 'Afghan hound',
 'basset',
 'beagle',
 'bloodhound',
 'bluetick',
 'black and tan coonhound',
 'Walker hound',
 'English foxhound',
 'redbone',
 'borzoi',
 'Irish wolfhound',
 'Italian greyhound',
 'whippet',
 'Ibizan hound',
 'Norwegian elkhound',
 'otterhound',
 'Saluki',
 'Scottish deerhound',
 'Weimaraner',
 'Staffordshire bullterrier',
 'American Staffordshire terrier',
 'Bedlington terrier',
 'Border terrier',
 'Kerry blue terrier',
 'Irish terrier',
 'Norfolk terrier',
 'Norwich terrier',
 'Yorkshire terrier',
 'wire haired fox terrier',
 'Lakeland terrier',
 'Sealyham terrier',
 'Airedale',
 'cairn',
 'Australian terrier',
 'Dandie Dinmont',
 'Boston bull',
 'miniature schnauzer',
 'giant schnauzer',
 'standard schnauzer',
 'Scotch terrier',
 'Tibetan terrier',
 'silky terrier',
 'soft coated wheaten terrier',
 'West Highland white terrier',
 'Lhasa',
 'flat coated retriever',
 'curly coated retriever',
 '골든리트리버', # 골든리트리버
 'Labrador retriever',
 'Chesapeake Bay retriever',
 'German short haired pointer',
 'vizsla',
 'English setter',
 'Irish setter',
 'Gordon setter',
 'Brittany spaniel',
 'clumber',
 'English springer',
 'Welsh springer spaniel',
 'cocker spaniel',
 'Sussex spaniel',
 'Irish water spaniel',
 'kuvasz',
 'schipperke',
 'groenendael',
 'malinois',
 'briard',
 'kelpie',
 'komondor',
 'Old English sheepdog',
 'Shetland sheepdog',
 'collie',
 'Border collie',
 'Bouvier des Flandres',
 'Rottweiler',
 'German shepherd',
 'Doberman',
 'miniature pinscher',
 'Greater Swiss Mountain dog',
 'Bernese mountain dog',
 'Appenzeller',
 'EntleBucher',
 'boxer',
 'bull mastiff',
 'Tibetan mastiff',
 'French bulldog',
 'Great Dane',
 'Saint Bernard',
 'Eskimo dog',
 'malamute',
 'Siberian husky',
 'affenpinscher',
 'basenji',
 'pug',
 'Leonberg',
 'Newfoundland',
 'Great Pyrenees',
 'Samoyed',
 'Pomeranian',
 'chow',
 'keeshond',
 'Brabancon griffon',
 'Pembroke',
 'Cardigan',
 'toy poodle',
 'miniature poodle',
 'standard poodle',
 'Mexican hairless',
 'dingo',
 'dhole',
 'African hunting dog']

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
    max_val, kls = torch.max(predictions, dim=0)
    print('Predicted :', breeds[kls])
    # plt.imshow(img.permute(1,2,0))
    # plt.show()
    return breeds[kls]


model = DogBreedPretrainedWideResnet()
weights_fname = 'classification/dog-breed-classifier-wideresnet_with_data_aug.pth'
# torch.save(model.state_dict(), weights_fname)
model.load_state_dict(torch.load(weights_fname, map_location=torch.device('cpu')))
model.eval()

device = get_default_device()
to_device(model, device)

app = Flask(__name__)
@app.route('/inference', methods=['POST'])
def inference():
    data = request.json
    numpy_data = np.array(data['images'], dtype=np.uint8)
    img = Image.fromarray(numpy_data)
    result = predict_new(img, model, device)
    return str(result)