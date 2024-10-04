from torch.utils.data import Dataset
import pandas as pd
from torchvision.io import read_image
import torch
import torch.nn as nn
from PIL import Image
from torchvision.transforms import v2



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CalfFaceDataset(Dataset):
    def __init__(self, df, image_dir, transform=None):
        self.data_frame = df
        self.transform = transform
        self.image_dir = image_dir

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        img_name = self.data_frame.loc[idx, "path"]  
        img_path = self.image_dir + '/' + img_name
        image = read_image(img_path) 
        label = self.data_frame.loc[idx, "target"]

        # Apply transformations if provided
        if self.transform:
            image = self.transform(image)

        # sample = {'image': image, 'bbox': (xmin, ymin, xmax, ymax), 'label': label}
        image = image.to(device)
        label = torch.tensor(label).to(device)

        sample = {'image': image, 'label': label, "img_path":img_path }
        return sample
        
    def get_labels(self):
        return self.data_frame["label"]


class CalfCenterFaceDataset(Dataset):
    def __init__(self, df, bbox_size = 800, transform=None):
        self.data_frame = df
        self.transform = transform
        self.bbox_size = bbox_size
        self.labels = self.data_frame["target"]
        
        label_to_count = self.labels.value_counts()

        weights = 1.0 / label_to_count[self.labels]

        self.weights = torch.DoubleTensor(weights.to_list())
        
    def __len__(self):
        return len(self.data_frame)

    def get_labels(self):
        return self.labels

    def get_class_weights(self):
        return torch.FloatTensor(1.0 / self.labels.value_counts())

    def __getitem__(self, idx):
        row = self.data_frame.loc[idx]
        img_path = self.data_frame.loc[idx, "path"]  
        image = Image.open(img_path)
        label = self.data_frame.loc[idx, "target"]

        xmin = row['x_min']
        ymin = row['y_min']
        xmax = row['x_max']
        ymax = row['y_max']
        
        # Calculate the width and height of the bounding box
        bbox_width = xmax - xmin
        bbox_height = ymax - ymin
        
        # Calculate the center of the first bounding box
        center_x = xmin + bbox_width / 2
        center_y = ymin + bbox_height / 2
        
        # Define the width and height of the second bounding box
        second_bbox_width = self.bbox_size
        second_bbox_height = self.bbox_size
        
        # Calculate the top-left corner of the second bounding box
        second_xmin = center_x - second_bbox_width / 2
        second_ymin = center_y - second_bbox_height / 2

        # Open the image

        # Crop the first bounding box from the image
        bbox_cropped_image = image.crop((xmin, ymin, xmax, ymax))
        
        # Resize the first cropped image to 800x800
        # first_resized_image = first_cropped_image.resize((self.bbox_size, self.bbox_size))
        
        # Crop the second bounding box from the image
        # img = Image.fromarray(img.cpu().permute(1, 2, 0).numpy())
        # image = image.crop((second_xmin, second_ymin, second_xmin + second_bbox_width, second_ymin + second_bbox_height))

        # img = v2.ToTensor()(img)
        image = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])(image)
        bbox_cropped_image = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])(bbox_cropped_image)
        
        # Apply transformations if provided
        if self.transform:
            image = self.transform(image)
            bbox_cropped_image = self.transform(bbox_cropped_image)

        # sample = {'image': image, 'bbox': (xmin, ymin, xmax, ymax), 'label': label}
        # image = image.to(device)
        bbox_cropped_image = bbox_cropped_image.to(device)
        label = torch.tensor(label, dtype=torch.float32).to(device)

        sample = {"row_image":image,  'image': bbox_cropped_image, 'label': label, "img_path":img_path, "weight": self.weights[idx] }
        return sample