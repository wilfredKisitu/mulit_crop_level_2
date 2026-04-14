import torch 
import torch.nn as nn

"""
    TODO: Check the case for more layers in each head for extraction 
    of other features
"""

class DiseaseHeads(nn.Module):
    """Implements multiple heads for direction of the model during training"""

    def __init__(self, feature_dim, disease_classes_per_plant):
        super().__init__()

        self.heads = nn.ModuleDict({
            "cassava": nn.Linear(feature_dim, disease_classes_per_plant['cassava']),
            "apple": nn.Linear(feature_dim, disease_classes_per_plant["apple"]),
            "banana": nn.Linear(feature_dim, disease_classes_per_plant["banana"]),
            "tomato": nn.Linear(feature_dim, disease_classes_per_plant["tomato"]),
            "maize": nn.Linear(feature_dim, disease_classes_per_plant["maize"])
        })
    
    def forward(self, features, plant_name):
        return self.heads[plant_name](features)
    

if __name__ == "__main__":
    """Implement the test case of the following model"""