import os
import pandas as pd
from torchvision import datasets, transforms, models
import torch
from torchvision.models import resnet50, ResNet50_Weights
from torch import nn, optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


# Path ai dati
DATASET_PATH = "./data"  # Sostituisci con il path del dataset di immagini
POKEDEX_PATH = "Pokedex.xlsx"  # Il file caricato


# Preprocessing dei dati
from torch.utils.data import random_split

# Preprocessing dei dati
def preprocess_data():
    # Trasformazioni delle immagini
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    dataset = datasets.ImageFolder(DATASET_PATH, transform=transform)
    
    # Dividi il dataset in train e test
    train_size = int(0.8 * len(dataset))  # 80% per il training
    test_size = len(dataset) - train_size
    train_data, test_data = random_split(dataset, [train_size, test_size])
    
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=32, shuffle=False)
    
    return train_loader, test_loader, dataset.classes

# Creazione e training del modello
def train_model_resnet50(train_loader, test_loader, class_names):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)  # ResNet50 preaddestrato
    model.fc = nn.Linear(model.fc.in_features, len(class_names))  # Adattiamo l'output al numero di classi
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=0.001)
    
    epochs = 10
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            # Aggiorna la loss
            running_loss += loss.item()
            
            # Calcolo dell'accuracy
            _, predicted = torch.max(outputs, 1)  # Ottieni le classi previste
            correct_predictions += (predicted == labels).sum().item()
            total_samples += labels.size(0)
        
        # Calcolo della loss media e dell'accuracy
        epoch_loss = running_loss / len(train_loader)
        epoch_accuracy = correct_predictions / total_samples
        
        print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}")

    
    # Valutazione sul test set
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    print(f"Accuracy: {100 * correct / total:.2f}%")
    
    # Salvataggio del modello
    torch.save(model.state_dict(), "pokemon_classifier.pth")
    return model

# Main script
if __name__ == "__main__":
    train_loader, test_loader, class_names = preprocess_data()
    train_model_resnet50(train_loader, test_loader, class_names)
