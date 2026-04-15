import torch
import torch.nn as nn

class PlantClassifier(nn.Module):

    def __init__(self, backbone, num_plants):
        super().__init__()
        self.backbone = backbone
        self.fc = nn.Linear(512, num_plants)

    
    def forward(self, x):
        features = self.backbone(x)
        plant_logits = self.fc(features)
        return plant_logits, features
    



if __name__ == '__main__':
    """Testing loading of the backend and output shape of the plant classifer"""

    from models.Resent_model import Resnet
    from Dataset.dataset_obj import PlantDataset
    from configs.project_dirs import TRAIN_PATH

    dataset = PlantDataset(TRAIN_PATH)
    num_plants = len(dataset.crop_types)
    num_diseases = len(dataset.disease_types)
    
    backbone = Resnet(num_plants, num_diseases, return_features=True)
    model = PlantClassifier(backbone, num_plants)

    model.eval()
    x  = torch.randn(4, 3, 32, 32)
    with torch.no_grad():
        plant_logits, features = model(x)

    print(f'Feature shape: {features.shape}')
    print(f'Plant logits shape: {plant_logits.shape}')