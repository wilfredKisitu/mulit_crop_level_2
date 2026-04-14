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
    

