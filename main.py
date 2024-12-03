import os
import pandas as pd
from torchvision import datasets, transforms, models
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


# Path ai dati
DATASET_PATH = "data"  # Sostituisci con il path del dataset di immagini
POKEDEX_PATH = "Pokedex.xlsx"  # Il file caricato

# Preprocessing dei dati
def preprocess_data():
    # Trasformazioni delle immagini
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    dataset = datasets.ImageFolder(DATASET_PATH, transform=transform)
    train_data, test_data = train_test_split(dataset, test_size=0.2, random_state=42)
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=32, shuffle=False)
    
    return train_loader, test_loader, dataset.classes

# Creazione e training del modello
def train_model(train_loader, test_loader, class_names):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.resnet18(pretrained=True)  # ResNet18 preaddestrato
    model.fc = nn.Linear(model.fc.in_features, len(class_names))  # Adattiamo l'output al numero di classi
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=0.001)
    
    epochs = 10
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(train_loader)}")
    
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
    train_model(train_loader, test_loader, class_names)
