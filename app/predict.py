import torch
from torchvision import models, transforms
from PIL import Image
import json

def load_image(image_path):
    image = Image.open(image_path).convert("RGB") # Ensure it's in RGB mode
preprocess = transforms.Compose([
transforms.Resize(256),
transforms.CenterCrop(224),
transforms.ToTensor(),
transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
image = preprocess(image).unsqueeze(0)
return image

def load_labels(json_path):
    with open(json_path) as f:
        labels = json.load(f)
# Check if the labels are in a list format and convert to a dictionary
if isinstance(labels, list):
    labels = {i: label for i, label in enumerate(labels)}
else: # Convert keys to integers if they are strings
    labels = {int(key): value for key, value in labels.items()}
return labels

def predict(image_path, model, labels):
    image = load_image(image_path)
model.eval()
with torch.no_grad():
    outputs = model(image)
_, predicted = torch.max(outputs, 1)
return labels[predicted.item()]

image_path = '/sdcard/IERG4998/cat.png'


model = models.resnet50(weights='IMAGENET1K_V1')

labels_path = '/sdcard/IERG4998/imagenet_labels.json' # Update this path
labels = load_labels(labels_path)

prediction = predict(image_path, model, labels)
print(f'Predicted grocery item: {prediction}')