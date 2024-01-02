from torch.utils.data import DataLoader
from CustomDataset import CustomDataset
from torchvision import transforms
import torch.nn as nn
from torch.optim import optimizer

list_of_samples = [
    {'image_path': 'dataset/images/image1.jpeg', 'label_path': 'dataset/labels/label1.txt'},
    {'image_path': 'dataset/images/image2.png', 'label_path': 'dataset/labels/label2.txt'},
    {'image_path': 'dataset/images/image3.jpeg', 'label_path': 'dataset/labels/label3.txt'},
    {'image_path': 'dataset/images/image4.jpeg', 'label_path': 'dataset/labels/label4.txt'},
    {'image_path': 'dataset/images/image5.png', 'label_path': 'dataset/labels/label5.txt'}
    # ...
]

# Example of using the dataset
transform = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])

custom_dataset = CustomDataset(list_of_samples, transform=transform)
data_loader = DataLoader(custom_dataset, batch_size=32, shuffle=True)
batch_size = 64
learning_rate = 0.001
num_epochs = 500
criterion = nn.MSELoss()
for epoch in range(num_epochs):
    for batch in data_loader:
        # Access batch['image'] and batch['label'] for training
        images = batch['image_path']
        labels = batch['label_path']

        # Your training code here
        # For example, you can perform forward and backward pass, update parameters, etc.
        # Replace the following lines with your actual training logic
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    # Print some information after each epoch
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')