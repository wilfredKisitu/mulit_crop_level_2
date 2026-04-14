import torch
import os
from models.Resent_model import Resnet
from Dataset.dataloader import PlantDataLoader
from Dataset.dataset_obj import PlantDataset
from sklearn.metrics import f1_score
import logging
import traceback


def train_one_epoch(model, dataloader, optimizer, loss_fn, device):
    """Trains pipeline which all computes the metrices for model evaluation"""

    model.train()

    total_loss = 0
    total_correct_plant = 0
    total_corret_disease = 0
    total_samples = 0

    all_plant_preds = []
    all_plant_labels = []

    all_disease_preds =[]
    all_disease_labeles = []

    for images, plant_label, disease_label in dataloader:
        images = images.to(device)
        plant_label = plant_label.to(device)
        disease_label = disease_label.to(device)

        plant_logits, disease_logits = model(images)
        loss_plant = loss_fn(plant_logits, plant_label)
        loss_disease = loss_fn(disease_logits, disease_label)

        loss  = loss_plant + loss_disease

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        plant_preds = torch.argmax(plant_logits, dim=1)
        disease_preds = torch.argmax(disease_logits, dim=1)

        total_correct_plant += (plant_preds == plant_label).sum().item()
        total_corret_disease += (disease_preds == disease_label).sum().item()
        total_samples += plant_label.size(0)

        all_plant_preds.extend(plant_preds.cpu().numpy())
        all_plant_labels.extend(plant_label.cpu().numpy())

        all_disease_preds.extend(disease_preds.cpu().numpy())
        all_disease_labeles.extend(disease_label.cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    plant_acc = total_correct_plant / total_samples
    disease_acc = total_corret_disease / total_samples

    plant_f1 = f1_score(all_plant_labels, all_plant_preds, average='weighted')
    disease_f1 = f1_score(all_disease_labeles, all_disease_preds, average='weighted')

    return avg_loss , plant_acc, disease_acc, plant_f1, disease_f1


def validate(model, dataloader, loss_fn, device):
    """Validation pipeline which all computes the metrices of the model"""

    model.eval()

    total_loss = 0
    total_correct_plant = 0
    total_correct_disease = 0

    all_plant_preds = []
    all_plant_labels = []

    all_disease_preds =[]
    all_disease_labeles = []

    total_samples = 0

    with torch.no_grad():
        for images, plant_labels, disease_labels in dataloader:
            images = images.to(device)
            plant_labels = plant_labels.to(device)
            disease_labels = disease_labels.to(device)

            plant_logit, disease_logits = model(images)

            loss_plant = loss_fn(plant_logit, plant_labels)
            loss_disease = loss_fn(disease_logits, disease_labels)
            loss = loss_plant + loss_disease

            total_loss += loss.item()
            plant_preds = torch.argmax(plant_logit, dim=1)
            disease_preds = torch.argmax(disease_logits, dim=1)

            total_correct_plant += (plant_preds == plant_labels).sum().item()
            total_correct_disease += (disease_preds == disease_labels).sum().item()

            total_samples += plant_labels.size(0)

            all_plant_preds.extend(plant_preds.cpu().numpy())
            all_plant_labels.extend(plant_labels.cpu().numpy())

            all_disease_preds.extend(disease_preds.cpu().numpy())
            all_disease_labeles.extend(disease_labels.cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    plant_acc = total_correct_plant / total_samples
    disease_acc = total_correct_disease / total_samples

    plant_f1 = f1_score(all_plant_labels, all_plant_preds, average='weighted')
    disease_f1 = f1_score(all_disease_labeles, all_disease_preds, average='weighted')

    return avg_loss, plant_acc, disease_acc, plant_f1, disease_f1


def train(model, train_loader, val_loader, optimizer, loss_fn, device, epochs= 10):
    """Combines the training loop"""
    os.system('clear')

    state_obj = {}
    best_val_loss = float('inf')

    for epoch in range(epochs):
        train_loss, train_plant_acc, train_disease_acc, train_plant_f1, train_disease_f1 = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
        

        val_loss, val_plant_acc, val_disease_acc, val_plant_f1, val_disease_f1 = validate(model, val_loader, loss_fn, device)

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
            }, 'best_model.pt')
            
        
        print(f"Epoch {epoch+1}/{epochs}")
        print(f"Train Loss: {train_loss:.4f} | Train Plant Acc: {train_plant_acc:.4f} | Train Disease acc: {train_disease_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f} | Val Plan Acc: {val_plant_acc:.4f} | Val Disease Acc: {val_disease_acc:.4f}")
        print("-"*50)

    torch.save(state_obj, "training_state.pt")
    

logging.basicConfig(
    filename="training.log",
    level = logging.INFO,
    format= "%(asctime)s - %(levelnames)s - %(message)s"
)

if __name__ == "__main__":
    try:
        ROOT_PATH ='/deepstore/datasets/dmb/ComputerVision/biology'
        TRAIN_PATH = os.path.join(ROOT_PATH, 'train-V')
        VAL_PATH = os.path.join(ROOT_PATH, 'test-V')
        TEST_PATH = os.path.join(ROOT_PATH, 'testing7')
        BATACH_SIZE = 32

        print(TRAIN_PATH, VAL_PATH, TEST_PATH, sep='\n')
        
        dataset = PlantDataset(TRAIN_PATH, is_test=False)
        dataloader = PlantDataLoader(dataset, batch_size=BATACH_SIZE, random=True)

        val_dataset = PlantDataset(VAL_PATH, is_test=False)
        val_dataloader = PlantDataLoader(val_dataset, batch_size=BATACH_SIZE)

        num_plants, num_diseases = len(dataset.crop_types), len(dataset.disease_types)

        model = Resnet(num_plants, num_diseases)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        loss_fn = torch.nn.CrossEntropyLoss()

        train( model, dataloader, val_dataloader, optimizer, loss_fn, device, epochs=1)

    except Exception as log_error:
        logging.error("Training failed")
        logging.error(traceback.format_exc())

        print('Error occurred. Check training.log')
        raise
