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
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Path ai dati
DATASET_PATH = "data"  # Sostituisci con il path del dataset di immagini

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
    
    #train_data, test_data = train_test_split(dataset, test_size=0.2, random_state=42)

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
    
    # Salvataggio del modello
    torch.save(model.state_dict(), "pokemon_classifier.pth")
    torch.save(model, 'entire_pokemon_classifier.pth')
    return model


# Funzione per calcolare e visualizzare la Confusion Matrix
def evaluate_model_with_confusion_matrix(model, test_loader, class_names):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            
            # Salva i label reali e le predizioni
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

    # Calcolo della confusion matrix
    cm = confusion_matrix(all_labels, all_predictions)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap="viridis", xticks_rotation='vertical')
    disp.ax_.set_title("Confusion Matrix")
    return cm

# Main script
if __name__ == "__main__":
    train_loader, test_loader, class_names = preprocess_data()
    model = train_model_resnet50(train_loader, test_loader, class_names)
    cm = evaluate_model_with_confusion_matrix(model, test_loader, class_names)
