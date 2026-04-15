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
            0: "apple",
            1: "bellpepper",
            2: "cherry",
            3: "corn",
            4: "grape",
            5: "peach",
            6: "potato",
            7: "strawberry",
            8: "tomato"
        }

    def forward(self, x, use_gt_plant= None):
        features = self.backbone(x)
        plant_logits = self.plant_head(features)

        if use_gt_plant is not None:
            plant_idx = use_gt_plant
        else:
            plant_idx = torch.argmax(plant_logits, dim=1)
        
        plant_name = [self.plant_idx_to_name[i.item()] for i in plant_idx]
        disease_logits = []
        
        for i, name in enumerate(plant_name):
            logits = self.disease_heads(features[i].unsqueeze(0), name)
            disease_logits.append(logits)
        disease_logits = torch.cat(disease_logits, dim=0)

        return plant_logits, disease_logits, features


if __name__ == "__main__":
    print(f"Testing the hierarchical model")
    import torch
    from Dataset.dataset_obj import PlantDataset
    from configs.project_dirs import TRAIN_PATH
    from models.Resent_model import Resnet

    dataset = PlantDataset(TRAIN_PATH)
    crops = dataset.crop_types
    num_plants = len(dataset.crop_types)
    num_diseases = len(dataset.disease_types)
    disease_class_per_plant = dataset.get_disease_per_crop_count()

    backbone = Resnet(num_plants, num_diseases, return_features=True)

    model = HierarchicalModel(
        backbone=backbone, 
        num_plantts=num_plants, 
        disease_class_per_plant=disease_class_per_plant
    )

    model.eval()
    print(f'Model initailized successfully')

    batch_size = 4
    x = torch.randn(batch_size, 3, 224, 224)
    plant_logits, disease_logits, features = model(x)

    feature_dim = 512

    print(f'Features shaped:  {features.shape}, expected shape: ({batch_size}, {feature_dim})')
    print(f'Plant logits shape: {plant_logits.shape}, expected shape: ({batch_size}, {num_plants})')
    print(f'Shape of the plant_logits: {plant_logits.shape}')
    

    print(f'Supported crops: \n {crops}')
