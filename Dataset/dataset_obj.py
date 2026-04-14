import os
import torch
from abc import ABC, abstractmethod
from PIL import Image
from torchvision import transforms

class Dataset(ABC):

    @abstractmethod
    def __len__(self):
        """computes  length of the obejct"""
        raise NotImplementedError('Child must provide implementation of this method')
    
    @abstractmethod
    def __getitem__(self, key):
        raise NotImplementedError(f'child must implement this method')
    

class PlantDataset(Dataset):
    """Loads the plant dataset and performs transformation on the data"""

    def __init__(self, root_path, is_test = False):
        self.root_path = root_path
        self.images = []
        self.is_test = is_test
        self.crop_types = []
        self.disease_types = []
        self._load_directories()
        

    def _load_directories(self):
        """Loads the train dataset by directory iteration"""
        for plant in os.listdir(self.root_path):
            plant_dir = os.path.join(self.root_path, plant)

            for cat in os.listdir(plant_dir):
                cat_dir = os.path.join(plant_dir, cat)
                for img in os.listdir(cat_dir):
                    img_dir = os.path.join(cat_dir, img)
                    if self.is_test:
                        for sub_f in os.listdir(img_dir):
                            sub_f_dir = os.path.join(img_dir, sub_f)
                    self.images.append(img_dir if not self.is_test else sub_f_dir)

                    # computes and loads the disease
                    plant_type, disease_type = self.get_label(img_dir if not self.is_test else sub_f_dir)
                    if plant_type not in self.crop_types:
                        self.crop_types.append(plant_type)
                    if disease_type not in self.disease_types:
                        self.disease_types.append(disease_type)

    def get_label(self, image_path):
        plant_type, disease_type = None, None
        path_splits= image_path.split('/')

        if self.is_test:
            plant_type, disease_type = path_splits[3], path_splits[-2]
        else:
            plant_type, disease_type = path_splits[2], path_splits[3]
        return plant_type, disease_type


    def __len__(self):
        """Computes the number of images in the dataset"""
        return len(self.images)
    
    def __getitem__(self, key):
        """Laods the images and the class label"""

        image_path = self.images[key]
        plant_type, disease_type = self.get_label(image_path)
        image = Image.open(image_path).convert("RGB")

        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
        img_tensor = transform(image)
        return img_tensor, torch.tensor(self.crop_types.index(plant_type), dtype=torch.long),\
              torch.tensor(self.disease_types.index(disease_type), dtype=torch.long)





if __name__ == "__main__":
    import os

    dataset = PlantDataset('./testmee', is_test=True)
    os.system('clear')

    print(f'Length of Dataset: {len(dataset)}')
    x, y_c, y_d = dataset[20]
    print(f'Image shape: {x.shape}')
    print(f'Plant type: {y_c}')
    print(f'Disease category: {y_d}')






