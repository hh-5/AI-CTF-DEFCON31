import base64
from io import BytesIO

import requests
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from PIL import Image
from torchvision.models import MobileNet_V2_Weights

# Load the MobileNetV2 model

model = torch.hub.load('pytorch/vision:v0.16.0', 'mobilenet_v2', weights=MobileNet_V2_Weights.DEFAULT)
model.eval()


# Set target class
target_class = 948

# Define preprocessing pipeline
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
])

# Load and preprocess the wolf image
image_path = "t_wolf.jpg"
image = Image.open(image_path)
input_tensor = preprocess(image)

# Convert tensor to uint8 for the attack
input_tensor_uint8 = (input_tensor * 255).clamp(0, 255).byte()

# Check if a GPU is available and if not, use a CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Move the model to the chosen device
model = model.to(device)

# Adjust the learning rate for uint8 attack
lr_uint8 = 255 * 0.1

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

# Adversarial attack in uint8 space
num_iterations = 100
optimizer = optim.SGD([input_tensor.requires_grad_()], lr=0.1)
criterion = nn.CrossEntropyLoss()

for iteration in range(num_iterations):
    # Convert uint8 tensor to float and set requires_grad
    input_tensor_float = input_tensor_uint8.float() / 255.0
    input_tensor_float.requires_grad_(True)

    # Normalize the float tensor
    input_tensor_normalized = normalize(input_tensor_float)

    input_batch = input_tensor_normalized.unsqueeze(0).to(device)
    outputs = model(input_batch)

    loss = criterion(outputs, torch.tensor([target_class]).to(device))
    loss.backward()

    # Scale the gradient for uint8 update
    grad_scaled = input_tensor_float.grad * lr_uint8

    # Update the uint8 tensor
    input_tensor_uint8 = (input_tensor_uint8.float() - grad_scaled).clamp(0, 255).byte()

    # Convert the perturbed uint8 tensor back to float for the next iteration
    input_tensor_float = input_tensor_uint8.float() / 255.0

# Denormalize the float tensor
denormalized_tensor = input_tensor_uint8.clone()
# for t, m, s in zip(denormalized_tensor, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]):
#     t = t.mul(s).add(m)
print (denormalized_tensor)
# Convert the denormalized tensor to uint8
output_image_uint8 = (denormalized_tensor * 255).clamp(0, 255).byte()

# Convert tensor to PIL Image and save
output_image = transforms.ToPILImage()(input_tensor_uint8.squeeze())
output_image.save("96apple.png")
output_image = transforms.ToPILImage()(input_tensor_uint8.squeeze())

# The original image dimensions
original_h, original_w = image.size

# Calculate padding
# For a 224x224 crop from a 256x256 resized image, the padding on each side is:
padding = (256 - 224) // 2

# Add padding to undo the center crop
padded_image = transforms.Pad(padding)(output_image)

# Resize to undo the original resize
resized_image = padded_image.resize((original_w, original_h))

# Save the image
resized_image.save("96apple.png")
# Print top probabilities and classes
with torch.no_grad():
    outputs = model(input_batch)
    probabilities = nn.Softmax(dim=1)(outputs)
    top_probs, top_classes = torch.topk(probabilities, 5)
    print("Top 5 probabilities:", top_probs.cpu().numpy().flatten())
    print("Top 5 classes:", top_classes.cpu().numpy().flatten())


def evaluate_image(image_path, model):
    # Preprocessing pipeline
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Load and preprocess the image
    image = Image.open(image_path)
    input_tensor = preprocess(image)
    input_batch = input_tensor.unsqueeze(0).to(device)

    # Get model predictions
    with torch.no_grad():
        outputs = model(input_batch)
        probabilities = nn.Softmax(dim=1)(outputs)
        top_probs, top_classes = torch.topk(probabilities, 5)
        print("Top 5 probabilities:", top_probs.cpu().numpy().flatten())
        print("Top 5 classes:", top_classes.cpu().numpy().flatten())


# Example usage
evaluate_image("96apple.png", model)

# with open('96apple.png', 'rb') as f:
#     input_data = base64.b64encode(f.read())
#
# def query(input_data):
#     response = requests.post('http://granny-jpg.advml.com/score', json={'data': input_data.decode('utf-8')})
#     return response.json()
#
# print(query(input_data))
