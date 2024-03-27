import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset, random_split, DataLoader
from PIL import Image
import os
from tqdm import tqdm
from preprocessing import train_loader

def train(model, trainloader, optimizer, criterion, device):
    model.train()
    print('Training')
    train_running_loss = 0.0
    train_running_correct = 0
    counter = 0
    for i, data in tqdm(enumerate(trainloader), total=len(trainloader)):
        counter += 1
        image, labels = data
        image = image.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(image)
        loss = criterion(outputs, labels)
        train_running_loss += loss.item()
        _, preds = torch.max(outputs.data, 1)
        train_running_correct += (preds == labels).sum().item()
        loss.backward()
        optimizer.step()

    epoch_loss = train_running_loss / counter
    epoch_acc = 100. * (train_running_correct / len(trainloader.dataset))
    return epoch_loss, epoch_acc

model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)

criterion = torch.nn.CrossEntropyLoss()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
model = model.to(device)
model_save_name = 'model_attachment_V1.pth'
path = F"/content/drive/MyDrive/Colab Notebooks/attachment/{model_save_name}"
torch.save(model.state_dict(), path)