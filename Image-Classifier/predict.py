import numpy as np
import argparse
import torch
from torch import nn, optim
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from torch.autograd import Variable
from collections import OrderedDict
import matplotlib.pyplot as plt
import json
from PIL import Image

parser = argparse.ArgumentParser()

parser.add_argument('input', default='./flowers/test/1/image_06752.jpg', nargs='?', action="store", type=str)
parser.add_argument('--dir', action="store", dest="data_dir", default="./flowers/")
parser.add_argument('checkpoint', default='./checkpointer.pth', nargs='?', action="store", type=str)
parser.add_argument('--top_k', default=5, dest="top_k", action="store", type=int)
parser.add_argument('--category_names', dest="category_names", action="store", default='cat_to_name.json')
parser.add_argument('--gpu', default="gpu", action="store", dest="gpu")

args = parser.parse_args()
path_image = args.input
number_of_outputs = args.top_k
device = args.gpu
json_name = args.category_names
path = args.checkpoint

def neuralnetwork_setup(architecture,dropout,hidden_units=409, learning_rate=0.001, device='gpu'):

    model = models.vgg19(pretrained= True)
    for parameter in model.parameters():
        parameter.requires_grad = False
    model.classifier = nn.Sequential(nn.Linear(model.classifier[0].in_features , hidden_units),
                                     nn.ReLU(),
                                     nn.Dropout(dropout),
                                     nn.Linear(hidden_units, 102),
                                     nn.LogSoftmax(dim=1))
    model = model.to('cuda')
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(),learning_rate)




    model.to('cuda')

    return model, criterion, optimizer

def load_checkpoint(path='checkpointer.pth'):
    checkpoint = torch.load(path)
    architecture = checkpoint['architecture']
    learning_rate = checkpoint['learning_rate']
    hidden_units = checkpoint['hidden_units']
    dropout = checkpoint['dropout']
    epochs = checkpoint['epochs']

    model,_,_ = neuralnetwork_setup(architecture,dropout,hidden_units,learning_rate)

    model.class_to_idx = checkpoint['class_to_idx']
    model.load_state_dict(checkpoint['state_dict'])

    return model

def process_image(image):
    image_PILIGRAM = Image.open(image)
    image_transformations = transforms.Compose([transforms.Resize(256),
                                         transforms.CenterCrop(224),
                                         transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406],
                                                              [0.229, 0.224, 0.225])])
    processed_image = image_transformations(image_PILIGRAM)
    return processed_image



def predict(image_path, model, topk=5):

    
    model = model.to('cuda')
    model.eval()

    img = process_image(image_path).numpy()
    img = torch.from_numpy(np.array([img])).float()


    with torch.no_grad():
        logps = model.forward(img.cuda())

    probabilities = torch.exp(logps).data
    top_probabilities = probabilities.topk(topk)

    return top_probabilities

def main():
    model=load_checkpoint(path)
    with open(json_name, 'r') as json_file:
        name = json.load(json_file)
        
    probabilitiessection = predict(path_image, model, number_of_outputs)
    probability = np.array(probabilitiessection[0][0])
    labels = [name[str(index + 1)] for index in np.array(probabilitiessection[1][0])]
    
    i = 0
    while i < number_of_outputs:
        print("{} Got a probability of {}".format(labels[i], probability[i]))
        i += 1
    print("Prediction over")

    
if __name__== "__main__":
    main()