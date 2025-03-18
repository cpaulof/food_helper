import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import csv

class Nutrition5kDataset(Dataset):
    def __init__(self, data_dir, split='train', transform=None):
        self.data_dir = data_dir
        self.split = split
        self.transform = transform

        if split == 'train':
            split_file = os.path.join(data_dir, 'dish_ids', 'splits', 'rgb_train_ids.txt')
        elif split == 'test':
            split_file = os.path.join(data_dir, 'dish_ids', 'splits', 'rgb_test_ids.txt')
        else:
            raise ValueError("Invalid split. Choose 'train' or 'test'.")

        with open(split_file, 'r') as f:
            self.image_ids = [line.strip() for line in f]
            self.image_ids = [image_id for image_id in self.image_ids if os.path.exists(os.path.join(data_dir, 'realsense_overhead', image_id, 'rgb.png'))]
        
        #self.image_ids = os.listdir(os.path.join(data_dir, 'realsense_overhead'))
        #self.image_ids = [image_id for image_id in self.image_ids if os.path.exists(os.path.join(data_dir, 'realsense_overhead', image_id, 'rgb.png'))]
        print(len(self.image_ids))

    def __len__(self):
        return len(self.image_ids)

    def get_csv_line(self, image_id):
      csv_files = [
          "./food_dataset/metadata/dish_metadata_cafe1.csv",
          #"./food_dataset/metadata/dish_metadata_cafe2.csv"
      ]

      for csv_file in csv_files:
          if os.path.exists(csv_file):
              with open(csv_file, 'r', encoding='utf-8') as file:
                  reader = csv.reader(file) 
                  for row in reader:
                      if row and row[0] == image_id:  
                          return row  
      return None  

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        image_path = os.path.join(self.data_dir, 'realsense_overhead', image_id, 'rgb.png')
        
        try:
          image = Image.open(image_path).convert('RGB')
        except FileNotFoundError:
          print(f"Warning: Image file not found at {image_path}. Skipping.")
          # Return a placeholder or handle the error as needed
          return torch.zeros(size=((3, 224, 224))), None

        if self.transform:
            image = self.transform(image)
        
        target = ",".join(self.get_csv_line(image_id))
        if target is None:
            return torch.zeros(size=((3, 224, 224))), None
        return image, target
      
norm_image = lambda x: x*2-1.
# Example Usage
data_dir = './food_dataset'
transform_train = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    transforms.Lambda(norm_image),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(degrees=20),
    # transforms.ColorJitter(brightness=0.001)
])
transform_test = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Lambda(norm_image),
    # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    # transforms.RandomHorizontalFlip(),
    # transforms.RandomRotation(degrees=10),
    # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)
])

train_dataset = Nutrition5kDataset(data_dir, split='train', transform=transform_train)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2) # Adjust num_workers as needed


test_dataset = Nutrition5kDataset(data_dir, split='test', transform=transform_test)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=2) # Adjust num_workers as needed
