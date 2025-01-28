import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import random

class PotatoLeafDataset(Dataset):
    """
    Custom Dataset for Potato Leaf Image Translation for Multi-Domain CycleGAN.
    """

    def __init__(self, root_dir, domains, transform=None):
        """
        Args:
            root_dir (string): Directory with all the potato leaf images.
            domains (list): List of domain names (e.g., ['healthy', 'early_blight', 'late_blight']).
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.domains = domains
        self.transform = transform

        # Create a dictionary that maps domains to their image paths
        self.images = {}
        for domain in domains:
            domain_dir = os.path.join(root_dir, domain)
            self.images[domain] = []
            for file_name in os.listdir(domain_dir):
                if file_name.endswith('.jpg') or file_name.endswith('.png'):
                    self.images[domain].append(os.path.join(domain_dir, file_name))

    def __len__(self):
        """
        Returns the number of samples available across all domains.
        """
        # Return the number of images in the first domain (all domains are assumed to have equal number of images)
        return len(self.images[self.domains[0]])

    def __getitem__(self, idx):
        """
        Args:
            idx (int): Index of the sample.
        
        Returns:
            dict: A dictionary with the images from different domains.
        """
        # Randomly select one domain to use as the source
        source_domain = random.choice(self.domains)
        
        # Choose a corresponding image for the selected source domain
        source_image_path = self.images[source_domain][idx]
        source_image = Image.open(source_image_path).convert("RGB")
        
        # Randomly choose a target domain
        target_domain = random.choice([domain for domain in self.domains if domain != source_domain])
        
        # Choose a corresponding image for the target domain
        target_image_path = random.choice(self.images[target_domain])
        target_image = Image.open(target_image_path).convert("RGB")

        # Apply transformations if provided
        if self.transform:
            source_image = self.transform(source_image)
            target_image = self.transform(target_image)

        # Return both source and target images with their respective domain labels
        return {
            'source': source_image,
            'target': target_image,
            'source_domain': source_domain,
            'target_domain': target_domain
        }

# Define transforms (you can add more preprocessing as needed)
# transform = transforms.Compose([
#     transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
#     transforms.ToTensor(),
#     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # Normalization for CycleGAN
# ])

# # Example usage:
# # Create an instance of the dataset
# train_dataset = PotatoLeafDataset(root_dir=config.TRAIN_DATA_PATH, domains=config.DOMAINS, transform=transform)

# # Create a DataLoader for the dataset
# train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
