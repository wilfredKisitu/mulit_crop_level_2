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
            "apple": nn.Linear(feature_dim, disease_classes_per_plant["apple"]),
            "bellpepper": nn.Linear(feature_dim, disease_classes_per_plant["bellpepper"]),
            "cherry": nn.Linear(feature_dim, disease_classes_per_plant["cherry"]),
            "corn": nn.Linear(feature_dim, disease_classes_per_plant["corn"]),
            "grape": nn.Linear(feature_dim, disease_classes_per_plant["grape"]),
            "peach": nn.Linear(feature_dim, disease_classes_per_plant["peach"]),
            "potato": nn.Linear(feature_dim, disease_classes_per_plant["potato"]),
            "strawberry": nn.Linear(feature_dim, disease_classes_per_plant["strawberry"]),
            "tomato": nn.Linear(feature_dim, disease_classes_per_plant["tomato"])
        })
    
    def forward(self, features, plant_name):
        return self.heads[plant_name](features)
    

if __name__ == "__main__":
    """Implement the test case of the following model"""
    import torch
    from Dataset.dataset_obj import PlantDataset
    from configs.project_dirs import TRAIN_PATH
   

    dataset = PlantDataset(TRAIN_PATH)
    crops = dataset.crop_types
    disease_per_crop_count = dataset.get_disease_per_crop_count()
    feature_dim = 512
    model = DiseaseHeads(feature_dim, disease_per_crop_count)

    model.eval()
    batch_size = 8
    features = torch.randn(batch_size, feature_dim)

    for plant, num_classes in disease_per_crop_count.items():
        out = model(features, plant)

        print(f'\nTesting plant: {plant}')
        print(f'Expected output shape: ({batch_size}, {num_classes})')
        print(f'Actual shape output: {tuple(out.shape)}')
    

    print(f'Crop types supported for adjustment of heads:\n {crops}')
    print(f'Disease per crop type: \n {disease_per_crop_count}')
    
