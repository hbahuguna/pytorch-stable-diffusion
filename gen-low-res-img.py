import torch
import torch.nn as nn
from torch.optim import optimizer
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image

from CustomDataset import CustomDataset
from DataLoader import batch

list_of_samples = [
    {'image_path': 'dataset/images/image1.jpeg', 'label_path': 'dataset/labels/label1.txt'},
    {'image_path': 'dataset/images/image2.png', 'label_path': 'dataset/labels/label2.txt'},
    {'image_path': 'dataset/images/image3.jpeg', 'label_path': 'dataset/labels/label3.txt'},
    {'image_path': 'dataset/images/image4.jpeg', 'label_path': 'dataset/labels/label4.txt'},
    {'image_path': 'dataset/images/image5.png', 'label_path': 'dataset/labels/label5.txt'}
    # ...
]
# Define a more complex generator network
class Generator(nn.Module):
    def __init__(self, input_size, output_channels, resolution):
        super(Generator, self).__init__()
        self.fc1 = nn.Linear(input_size, 512)
        self.fc2 = nn.Linear(512, resolution * resolution * output_channels)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = x.view(-1, output_channels, resolution, resolution)
        return x

# Define training parameters
batch_size = 64
learning_rate = 0.001
num_epochs = 500

resolution = 128  # Increase resolution for higher-quality images
seed_size = 100
output_channels = 1  # Grayscale image
# Initialize the generator
generator = Generator(seed_size, output_channels, resolution)

# Define a custom dataset and data loader (replace with your dataset)
dataset = CustomDataset(list_of_samples='dataset', transform=transforms.ToTensor())
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Generate a random seed for image generation
def generate_seed(seed_size):
    return torch.randn(seed_size)

# Generate a higher-resolution black and white image
def generate_image(generator, seed):
    with torch.no_grad():
        generator.eval()
        output = generator(seed)
        output = torch.sigmoid(output)
    return output

# Save the generated image
def save_image(image, save_path):
    image = transforms.ToPILImage()(image.squeeze(0))
    image.save(save_path)
    image.show()

if __name__ == "__main__":
    # Set random seed for reproducibility
    torch.manual_seed(42)

    # Define image parameters
    resolution = 128  # Increase resolution for higher-quality images
    seed_size = 100
    output_channels = 1  # Grayscale image

    # Initialize the generator
    generator = Generator(seed_size, output_channels, resolution)

    # Allow the user to enter a text prompt
    user_prompt = input("Enter a text prompt: ")

    # Generate a random seed
    seed = generate_seed(seed_size)

    # Train the model for more epochs (adjust as needed)
    num_epochs = 500
    # Define loss function and optimizer
    criterion = nn.MSELoss()
    for epoch in range(num_epochs):
        # Your training loop here (not provided in this example)
        inputs = batch['image_path']  # Replace 'images' with the key used in your dataset
        targets = batch['label_path']  # Replace 'labels' with the key used in your dataset

        # Flatten the inputs (assuming 2D images)
        inputs = inputs.view(inputs.size(0), -1)

        # Generate random seed
        seed = generate_seed(seed_size)

        # Forward pass
        outputs = generator(seed)

        # Calculate the loss
        loss = criterion(outputs, targets)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Print the loss for every epoch
    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

    # Generate and save the higher-resolution image based on the user's prompt
    generated_image = generate_image(generator, seed)
    save_image(generated_image, f"generated_image_{user_prompt}.png")