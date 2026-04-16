import os
import torch
from models.Hierarchical_model import HierarchicalModel
from models.Resent_model import Resnet
from Dataset.dataloader import PlantDataLoader
from Dataset.dataset_obj import PlantDataset
from model_train.train_hier import validate, build_disease_local_idx_map
from model_train.test import filter_valid_paths, ROOT_PATH, TEST_SPLITS, BATCH_SIZE
from configs.project_dirs import TRAIN_PATH


CHECKPOINT = 'best_hier_model.pt'

train_dataset = PlantDataset(TRAIN_PATH)
crop_types = train_dataset.crop_types
disease_types = train_dataset.disease_types

idx_to_crop = {i: crop for crop, i in train_dataset.crop_to_idx.items()}
disease_local_idx_map = build_disease_local_idx_map(train_dataset)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

backbone = Resnet(len(crop_types), len(disease_types), return_features=True)
model = HierarchicalModel(
    backbone=backbone,
    num_plantts=len(crop_types),
    disease_class_per_plant=train_dataset.get_disease_per_crop_count()
)
checkpoint = torch.load(CHECKPOINT, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model = model.to(device)

loss_fn = torch.nn.CrossEntropyLoss()

state_obj = {}

for split in TEST_SPLITS:
    test_path = os.path.join(ROOT_PATH, split)
    valid_paths = filter_valid_paths(test_path, crop_types, disease_types)
    test_dataset = PlantDataset(test_path, crop_types=crop_types, disease_types=disease_types, is_train=False, image_paths=valid_paths)
    test_loader = PlantDataLoader(test_dataset, batch_size=BATCH_SIZE)

    loss, plant_acc, disease_acc, plant_f1, disease_f1 = validate(model, test_loader, loss_fn, device, idx_to_crop, disease_local_idx_map)

    state_obj[split] = {
        'loss': loss,
        'plant_acc': plant_acc,
        'disease_acc': disease_acc,
        'plant_f1': plant_f1,
        'disease_f1': disease_f1,
    }

    print(f"[{split}] Loss: {loss:.4f} | Plant Acc: {plant_acc:.4f} | Plant F1: {plant_f1:.4f} | Disease Acc: {disease_acc:.4f} | Disease F1: {disease_f1:.4f}")

torch.save(state_obj, 'test_hier_state.pt')
