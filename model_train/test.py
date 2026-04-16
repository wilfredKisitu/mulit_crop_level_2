import os
import torch
from models.Resent_model import Resnet
from Dataset.dataloader import PlantDataLoader
from Dataset.dataset_obj import PlantDataset
from model_train.train import validate
from configs.project_dirs import TRAIN_PATH, TEST_PATH


BATCH_SIZE = 64
CHECKPOINT = 'best_model.pt'

ROOT_PATH = os.path.dirname(TRAIN_PATH)
TEST_SPLITS = ['testA', 'testB', 'testC', 'testD', 'testE', 'testF']


def filter_valid_paths(test_path, crop_types, disease_types):
    """Returns only image paths whose crop and disease labels exist in the training vocabulary."""
    crop_set = set(crop_types)
    disease_set = set(disease_types)
    valid = []
    for plant in os.listdir(test_path):
        if plant.lower() not in crop_set:
            continue
        plant_dir = os.path.join(test_path, plant)
        for disease in os.listdir(plant_dir):
            if disease.lower() not in disease_set:
                continue
            disease_dir = os.path.join(plant_dir, disease)
            for img in os.listdir(disease_dir):
                valid.append(os.path.join(disease_dir, img))
    return valid


if __name__ == "__main__":
    train_dataset = PlantDataset(TRAIN_PATH)
    crop_types = train_dataset.crop_types
    disease_types = train_dataset.disease_types

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    checkpoint = torch.load(CHECKPOINT, map_location=device)
    model = Resnet(len(crop_types), len(disease_types))
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)

    loss_fn = torch.nn.CrossEntropyLoss()

    state_obj = {}

    for split in TEST_SPLITS:
        test_path = os.path.join(TEST_PATH, split)
        valid_paths = filter_valid_paths(test_path, crop_types, disease_types)
        test_dataset = PlantDataset(test_path, crop_types=crop_types, disease_types=disease_types, is_train=False, image_paths=valid_paths)
        test_loader = PlantDataLoader(test_dataset, batch_size=BATCH_SIZE)

        loss, plant_acc, disease_acc, plant_f1, disease_f1 = validate(model, test_loader, loss_fn, device)

        state_obj[split] = {
            'loss': loss,
            'plant_acc': plant_acc,
            'disease_acc': disease_acc,
            'plant_f1': plant_f1,
            'disease_f1': disease_f1,
        }

        print(f"[{split}] Loss: {loss:.4f} | Plant Acc: {plant_acc:.4f} | Plant F1: {plant_f1:.4f} | Disease Acc: {disease_acc:.4f} | Disease F1: {disease_f1:.4f}")

    torch.save(state_obj, 'test_state.pt')
