import numpy as np
import os

import torch
from torch import nn
from torchvision import models, transforms
from collections import OrderedDict
from torch.autograd import Variable

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def ResNet18(imagenet_weights=True):
    model = models.resnet18(pretrained=imagenet_weights)
    model.fc = nn.Sequential(OrderedDict([
                            ('drop1', nn.Dropout(0.5)),
                            ('fc1', nn.Linear(512, 25)),
                            ('output', nn.LogSoftmax(dim=1))
                            ]))
    return model

def transform_function():
    image_transform = transforms.Compose([transforms.Resize((224,224)),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
                                      ])
    return image_transform

def preprocess_image(img):
    image_transform = transform_function()
    
    img_for_model = image_transform(img).float()
    img_for_model = Variable(img_for_model,requires_grad=True)
    img_for_model = img_for_model.unsqueeze(0).to(device)
    return img_for_model

def load_model(model_path):

    model = ResNet18(imagenet_weights=False)
    model.load_state_dict(torch.load(os.path.join(models_path, 'ResNet18.pth'), map_location=device))
    model.to(device)

    return model
   

def run_model(img, model):
    classes = ['aceite', 'agua', 'arroz', 'azucar', 'cafe', 'caramelo', 'cereal', 
           'chips', 'chocolate', 'especias', 'frijoles', 'gaseosa', 'harina', 
           'jamon', 'jugo', 'leche', 'maiz', 'miel', 'nueces', 'pasta', 
           'pescado', 'salsatomate', 'te', 'torta', 'vinagre']

    output_dict = {}

    with torch.no_grad():
        model.eval()
        output = torch.softmax(model(img), 1)
        score, class_predicted = torch.max(output.data, 1)
        output_dict = {'class': classes[class_predicted.item()], 'score': round(score.item(), 4)}
    
    return output_dict