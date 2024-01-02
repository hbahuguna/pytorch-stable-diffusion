from PIL import Image
from torch.utils.data import Dataset


class CustomDataset(Dataset):
    def __init__(self, list_of_samples, transform=None):
        self.list_of_samples = list_of_samples
        self.transform = transform

    def __len__(self):
        return len(self.list_of_samples)

    def __getitem__(self, idx):
        # Load image
        image_path = self.list_of_samples[idx]['image_path']
        image = Image.open(image_path).convert('RGB')

        # Load label (example assumes label is stored in a text file)
        label_path = self.list_of_samples[idx]['label_path']
        with open(label_path, 'r') as label_file:
            label = label_file.read().strip()

        # Apply transformation if specified
        if self.transform:
            image = self.transform(image)

        return {'image': image, 'label': label}
