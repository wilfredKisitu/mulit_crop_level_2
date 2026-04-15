import torch
import os
import logging
import traceback
from sklearn.metrics import f1_score

from models.Hierarchical_model import HierarchicalModel
from models.Resent_model import Resnet
from Dataset.dataloader import PlantDataLoader
from Dataset.dataset_obj import PlantDataset


def build_disease_local_idx_map(dataset):
    """
    Builds {crop_name: {global_disease_idx: local_disease_idx}}.
    Needed because each crop's disease head uses local indices (0..N-1),
    but the dataset returns global disease indices.
    """
    mapping = {}
    for crop, diseases in dataset.crop_disease_dict.items():
        mapping[crop] = {}
        for local_idx, disease_name in enumerate(diseases):
            global_idx = dataset.disease_to_idx[disease_name]
            mapping[crop][global_idx] = local_idx
    return mapping


def train_one_epoch(model, dataloader, optimizer, loss_fn, device, idx_to_crop, disease_local_idx_map):
    """Trains one epoch of the hierarchical model and computes metrics."""

    model.train()

    total_loss = 0
    total_correct_plant = 0
    total_correct_disease = 0
    total_samples = 0

    all_plant_preds = []
    all_plant_labels = []
    all_disease_preds = []
    all_disease_labels = []

    for images, plant_label, disease_label in dataloader:
        images = images.to(device)
        plant_label = plant_label.to(device)
        disease_label = disease_label.to(device)

        optimizer.zero_grad()

        # Backbone features and plant classification
        features = model.backbone(images)
        plant_logits = model.plant_head(features)
        loss_plant = loss_fn(plant_logits, plant_label)

        # Per-sample disease loss using ground truth plant labels (teacher forcing)
        disease_loss = torch.tensor(0.0, device=device)
        batch_disease_preds = []
        batch_disease_local_labels = []

        for i in range(images.size(0)):
            crop_name = idx_to_crop[plant_label[i].item()]
            local_dis_idx = disease_local_idx_map[crop_name][disease_label[i].item()]
            logits = model.disease_heads(features[i].unsqueeze(0), crop_name)
            local_label = torch.tensor([local_dis_idx], device=device)
            disease_loss += loss_fn(logits, local_label)
            batch_disease_preds.append(torch.argmax(logits, dim=1).item())
            batch_disease_local_labels.append(local_dis_idx)

        disease_loss = disease_loss / images.size(0)
        loss = loss_plant + disease_loss

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()

        plant_preds = torch.argmax(plant_logits, dim=1)
        total_correct_plant += (plant_preds == plant_label).sum().item()
        total_correct_disease += sum(p == l for p, l in zip(batch_disease_preds, batch_disease_local_labels))
        total_samples += plant_label.size(0)

        all_plant_preds.extend(plant_preds.cpu().numpy())
        all_plant_labels.extend(plant_label.cpu().numpy())
        all_disease_preds.extend(batch_disease_preds)
        all_disease_labels.extend(batch_disease_local_labels)

    avg_loss = total_loss / len(dataloader)
    plant_acc = total_correct_plant / total_samples
    disease_acc = total_correct_disease / total_samples
    plant_f1 = f1_score(all_plant_labels, all_plant_preds, average='weighted')
    disease_f1 = f1_score(all_disease_labels, all_disease_preds, average='weighted')

    return avg_loss, plant_acc, disease_acc, plant_f1, disease_f1


def validate(model, dataloader, loss_fn, device, idx_to_crop, disease_local_idx_map):
    """Validates the hierarchical model and computes metrics."""

    model.eval()

    total_loss = 0
    total_correct_plant = 0
    total_correct_disease = 0
    total_samples = 0

    all_plant_preds = []
    all_plant_labels = []
    all_disease_preds = []
    all_disease_labels = []

    with torch.no_grad():
        for images, plant_labels, disease_labels in dataloader:
            images = images.to(device)
            plant_labels = plant_labels.to(device)
            disease_labels = disease_labels.to(device)

            features = model.backbone(images)
            plant_logits = model.plant_head(features)
            loss_plant = loss_fn(plant_logits, plant_labels)

            disease_loss = torch.tensor(0.0, device=device)
            batch_disease_preds = []
            batch_disease_local_labels = []

            for i in range(images.size(0)):
                crop_name = idx_to_crop[plant_labels[i].item()]
                local_dis_idx = disease_local_idx_map[crop_name][disease_labels[i].item()]
                logits = model.disease_heads(features[i].unsqueeze(0), crop_name)
                local_label = torch.tensor([local_dis_idx], device=device)
                disease_loss += loss_fn(logits, local_label)
                batch_disease_preds.append(torch.argmax(logits, dim=1).item())
                batch_disease_local_labels.append(local_dis_idx)

            disease_loss = disease_loss / images.size(0)
            loss = loss_plant + disease_loss

            total_loss += loss.item()

            plant_preds = torch.argmax(plant_logits, dim=1)
            total_correct_plant += (plant_preds == plant_labels).sum().item()
            total_correct_disease += sum(p == l for p, l in zip(batch_disease_preds, batch_disease_local_labels))
            total_samples += plant_labels.size(0)

            all_plant_preds.extend(plant_preds.cpu().numpy())
            all_plant_labels.extend(plant_labels.cpu().numpy())
            all_disease_preds.extend(batch_disease_preds)
            all_disease_labels.extend(batch_disease_local_labels)

    avg_loss = total_loss / len(dataloader)
    plant_acc = total_correct_plant / total_samples
    disease_acc = total_correct_disease / total_samples
    plant_f1 = f1_score(all_plant_labels, all_plant_preds, average='weighted')
    disease_f1 = f1_score(all_disease_labels, all_disease_preds, average='weighted')

    return avg_loss, plant_acc, disease_acc, plant_f1, disease_f1


def train(model, train_loader, val_loader, optimizer, loss_fn, device, idx_to_crop, disease_local_idx_map, epochs=10):
    """Full training loop for the hierarchical model."""

    state_obj = {}
    best_val_loss = float('inf')

    for epoch in range(epochs):
        train_loss, train_plant_acc, train_disease_acc, train_plant_f1, train_disease_f1 = train_one_epoch(
            model, train_loader, optimizer, loss_fn, device, idx_to_crop, disease_local_idx_map
        )
        val_loss, val_plant_acc, val_disease_acc, val_plant_f1, val_disease_f1 = validate(
            model, val_loader, loss_fn, device, idx_to_crop, disease_local_idx_map
        )

        state_obj[epoch] = {
            'train_loss': train_loss,
            'train_plant_acc': train_plant_acc,
            'train_disease_acc': train_disease_acc,
            'train_plant_f1': train_plant_f1,
            'train_disease_f1': train_disease_f1,
            'val_loss': val_loss,
            'val_plant_acc': val_plant_acc,
            'val_disease_acc': val_disease_acc,
            'val_plant_f1': val_plant_f1,
            'val_disease_f1': val_disease_f1
        }

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_loss': val_loss
            }, 'best_hier_model.pt')

        print(f"Epoch {epoch+1}/{epochs}")
        print(f"Train Loss: {train_loss:.4f} | Train Plant Acc: {train_plant_acc:.4f} | Train Disease Acc: {train_disease_acc:.4f}")
        print(f"Val Loss:   {val_loss:.4f} | Val Plant Acc:   {val_plant_acc:.4f} | Val Disease Acc:   {val_disease_acc:.4f}")
        print("-" * 50)

        logging.info(
            f"Epoch {epoch+1}/{epochs} | "
            f"Train Loss: {train_loss:.4f} | Train Plant Acc: {train_plant_acc:.4f} | Train Disease Acc: {train_disease_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} | Val Plant Acc: {val_plant_acc:.4f} | Val Disease Acc: {val_disease_acc:.4f}"
        )

    torch.save(state_obj, "hier_training_state.pt")


if __name__ == "__main__":
    logging.basicConfig(
        filename="hier_training.log",
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

    try:
        ROOT_PATH = '/deepstore/datasets/dmb/ComputerVision/biology'
        TRAIN_PATH = os.path.join(ROOT_PATH, 'train-V')
        VAL_PATH = os.path.join(ROOT_PATH, 'test-V')
        BATCH_SIZE = 64

        print(TRAIN_PATH, VAL_PATH, sep='\n')

        dataset = PlantDataset(TRAIN_PATH)
        dataloader = PlantDataLoader(dataset, batch_size=BATCH_SIZE, random=True)

        global_crop_types = dataset.crop_types
        global_disease_types = dataset.disease_types

        val_dataset = PlantDataset(VAL_PATH, crop_types=global_crop_types, disease_types=global_disease_types, is_train=False)
        val_dataloader = PlantDataLoader(val_dataset, batch_size=BATCH_SIZE)

        num_plants = len(dataset.crop_types)
        num_diseases = len(dataset.disease_types)
        disease_class_per_plant = dataset.get_disease_per_crop_count()

        idx_to_crop = {i: crop for crop, i in dataset.crop_to_idx.items()}
        disease_local_idx_map = build_disease_local_idx_map(dataset)

        backbone = Resnet(num_plants, num_diseases, return_features=True)
        model = HierarchicalModel(
            backbone=backbone,
            num_plantts=num_plants,
            disease_class_per_plant=disease_class_per_plant
        )

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        loss_fn = torch.nn.CrossEntropyLoss()

        train(model, dataloader, val_dataloader, optimizer, loss_fn, device, idx_to_crop, disease_local_idx_map, epochs=20)

    except Exception as log_error:
        logging.error("Hierarchical training failed")
        logging.error(traceback.format_exc())
        print('Error occurred. Check hier_training.log')
        raise
