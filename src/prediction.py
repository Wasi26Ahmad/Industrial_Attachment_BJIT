import torch
from torchvision import transforms
from PIL import Image


def load_model():
    model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
    model.load_state_dict(torch.load('model_attachment_V1.pth.pth'))
    model.eval()
    return model

def predict_image(image_path, model):

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    image = Image.open(image_path)
    image = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)

    if predicted.item() == 0:
        return "Car"
    else:
        return "Bike"

if __name__ == "__main__":
    model = load_model()
    result = predict_image('path_to_your_image.jpg', model)
    print(result)




