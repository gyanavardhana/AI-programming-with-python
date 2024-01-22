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

parser.add_argument('data_dir', action="store", default="./flowers/")
parser.add_argument('--save_dir', action="store", default="./checkpointer.pth")
parser.add_argument('--arch', action="store", default="vgg19")
parser.add_argument('--learning_rate', action="store", type=float,default=0.001)
parser.add_argument('--hidden_units', action="store", dest="hidden_units", type=int, default=512)
parser.add_argument('--epochs', action="store", default=3, type=int)
parser.add_argument('--dropout', action="store", type=float, default=0.2)
parser.add_argument('--gpu', action="store", default="gpu")

args = parser.parse_args()
where = args.data_dir
path = args.save_dir
lr = args.learning_rate
architecture = args.arch
hidden_units = args.hidden_units
power = args.gpu
epochs = args.epochs
dropout = args.dropout


device = torch.device("cuda")


def load_data(mai = "./flowers"):
    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)
    
    data_dir = mai
    training_dir = data_dir + '/train'
    validating_dir = data_dir + '/valid'
    testing_dir = data_dir + '/test'

    training_transforms = transforms.Compose([transforms.RandomRotation(30),
                                              transforms.RandomResizedCrop(224),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.ToTensor(),
                                              transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),])

    testing_transforms = transforms.Compose([transforms.Resize(256),
                                             transforms.CenterCrop(224),
                                             transforms.ToTensor(),
                                             transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),])

    validating_transforms = transforms.Compose([transforms.Resize(256),
                                                transforms.CenterCrop(224),
                                                transforms.ToTensor(),
                                                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),])

    training_data = datasets.ImageFolder(training_dir, transform = training_transforms)
    testing_data = datasets.ImageFolder(testing_dir, transform = testing_transforms)
    validating_data = datasets.ImageFolder(validating_dir, transform = validating_transforms)


    trainingloader = torch.utils.data.DataLoader(training_data, batch_size = 64, shuffle = True)
    testingloader = torch.utils.data.DataLoader(testing_data, batch_size = 64, shuffle = True)
    validatingloader = torch.utils.data.DataLoader(validating_data, batch_size = 64, shuffle = True)

    return trainingloader, validatingloader, testingloader, training_data



def neuralnetwork_setup(architecture='vgg19',dropout=0.1,hidden_units=4096, learning_rate=0.001, device='gpu'):

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

def main():
    trainingloader, validatingloader, testingloader, training_data = load_data(where)
    model, criterion, optimizer = neuralnetwork_setup(architecture,dropout,hidden_units,lr,power)
    num_epochs = epochs
    print(num_epochs)
    print_interval = 5
    total_steps = 0
    loss_history = []

    epoch = 0
    while epoch < num_epochs:
        running_loss = 0
        step_count = 0

        while step_count < len(trainingloader):
            inputs, labels = next(iter(trainingloader))
            total_steps += 1
            step_count += 1
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if total_steps % print_interval == 0:
                model.eval()
                validation_loss = 0
                accuracy = 0

                with torch.no_grad():
                    for inputs, labels in validatingloader:
                        inputs, labels = inputs.to(device), labels.to(device)
                        log_ps = model.forward(inputs)
                        batch_loss = criterion(log_ps, labels)
                        validation_loss += batch_loss.item()

                        ps = torch.exp(log_ps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                print(f"Epoch {epoch + 1}/{num_epochs}.. "
                      f"Training Loss: {running_loss/print_interval:.3f}.. "
                      f"Validation Loss: {validation_loss/len(validatingloader):.3f}.. "
                      f"Accuracy: {accuracy/len(validatingloader):.3f}")

                running_loss = 0
                model.train()
        epoch += 1
    

    model.class_to_idx = training_data.class_to_idx
    torch.save({'architecture': architecture,
                'learning_rate': lr,
                'hidden_units': hidden_units,
                'dropout': dropout,
                'epochs': epochs,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'class_to_idx': model.class_to_idx},
                path)
    print("checkpoint noted!")
if __name__ == "__main__":
    main()