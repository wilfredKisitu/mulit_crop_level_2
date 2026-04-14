import torch
import torch.nn as nn
from models.disease_heads import DiseaseHeads

class HierarchicalModel(nn.Module):

    def __init__(self, backbone, num_plantts, disease_class_per_plant):
        super().__init__()
        self.backbone = backbone
        self.plant_head = nn.Linear(512, num_plantts)
        self.disease_heads= DiseaseHeads(512, disease_class_per_plant)

        self.plant_idx_to_name = {
            0: "cassava",
            1: "apple",
            2: "banana",
            3: "tomato",
            4: "maize"
        }

    def forward(self, x):
        features = self.backbone(x)
        plant_logits = self.plant_head(features)
        return plant_logits, features
    
