import os
import torch
from abc import ABC, abstractmethod
from PIL import Image
from torchvision import transforms
from collections import Counter


_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]

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

    def __init__(self, root_path, crop_types=None, disease_types=None, is_train=True):
        self.root_path = root_path
        self.images = []
        self._load_directories()

        if crop_types is None and disease_types is None:
            self.crop_types = sorted(self.crop_types)
            self.disease_types = sorted(self.disease_types)
        else:
            self.crop_types = crop_types
            self.disease_types = disease_types

        self.crop_to_idx = {c: i for i, c in enumerate(self.crop_types)}
        self.disease_to_idx = {d: i for i, d in enumerate(self.disease_types)}

        self._get_crop_disease_dict()
        self._get_disease_per_crop_count()

        if is_train:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(_IMAGENET_MEAN, _IMAGENET_STD)
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(_IMAGENET_MEAN, _IMAGENET_STD)
            ])

        

    def _load_directories(self):
        """Loads the train dataset by directory iteration"""
        self.crop_types = []
        self.disease_types = []

        for plant in os.listdir(self.root_path):
            plant_dir = os.path.join(self.root_path, plant)

            for cat in os.listdir(plant_dir):
                cat_dir = os.path.join(plant_dir, cat)
                for img in os.listdir(cat_dir):
                    img_dir = os.path.join(cat_dir, img)
                    self.images.append(img_dir)
                    plant_type, disease_type = self.get_label(img_dir)

                    if plant_type not in self.crop_types:
                        self.crop_types.append(plant_type)
                    if disease_type not in self.disease_types:
                        self.disease_types.append(disease_type)

    def _get_crop_disease_dict(self):
        """Builds a state of crops with their respective diseases"""
        crop_disease_dict = dict()
        for crop in self.crop_types:
            crop_disease_dict[crop] = list()
        
        for disease in self.disease_types:
            _crop_index = self.crop_types.index(disease.split('_')[0])
            _crop_key = self.crop_types[_crop_index]
            crop_disease_dict[_crop_key].append(disease)
        
        self.crop_disease_dict = crop_disease_dict

    
    def _get_disease_per_crop_count(self):
        disease_per_crop_counter = Counter({crop: len(disease) for crop, disease in self.crop_disease_dict.items()})
        
        self.disease_per_crop_counter = disease_per_crop_counter

    def get_disease_per_crop(self):
        """Returns diseases per crop"""
        return self.crop_disease_dict
    
    def get_disease_per_crop_count(self):
        """Return number of diseases per crop"""
        return self.disease_per_crop_counter
        

    def get_label(self, image_path):
        plant_type, disease_type = None, None
        path_splits= image_path.split('/')
        plant_type, disease_type = path_splits[-3], path_splits[-2]
        
        return plant_type, disease_type


    def __len__(self):
        """Computes the number of images in the dataset"""
        return len(self.images)
    
    def __getitem__(self, key):
        """Laods the images and the class label"""

        image_path = self.images[key]
        plant_type, disease_type = self.get_label(image_path)
        image = Image.open(image_path).convert("RGB")

        img_tensor = self.transform(image)
        crop_label = self.crop_to_idx[plant_type]
        disease_label = self.disease_to_idx[disease_type]

        return (
            img_tensor, 
            torch.tensor(crop_label, dtype=torch.long),
            torch.tensor(disease_label, dtype=torch.long)
        )




if __name__ == "__main__":
    import os
    from configs.project_dirs import TRAIN_PATH, TEST_PATH
    
    dataset = PlantDataset(TRAIN_PATH)
    os.system('clear')

    print(f'Length of Dataset: {len(dataset)}')
    x, y_c, y_d = dataset[20]
    num_of_crops = len(dataset.crop_types)
    num_of_diseases = len(dataset.disease_types)

    print(f'Image shape: {x.shape}')
    print(f'Plant type: {y_c}')
    print(f'Disease category: {y_d}')
    print(f'Num of crops: {num_of_crops}')
    print(f'Number of diseases: {num_of_diseases}')

    





