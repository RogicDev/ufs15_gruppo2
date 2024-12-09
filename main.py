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
from torch.utils.data import random_split
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, precision_recall_fscore_support
import seaborn as sns

# Path ai dati
DATASET_PATH = "data"  # Sostituisci con il path del dataset di immagini

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
    
    # Calcolo dei pesi per classe
    dataset = train_loader.dataset.dataset  # Ottieni il dataset di ImageFolder dall'oggetto random_split
    class_weights = calculate_class_weights(dataset).to(device)
    
    # Definisci la funzione di perdita con i pesi
    criterion = nn.CrossEntropyLoss(weight=class_weights)
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

    # Raccolta delle predizioni e dei label reali
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

    # Calcolo della confusion matrix
    cm = confusion_matrix(all_labels, all_predictions, normalize='true')

    # Heatmap con Seaborn
    plt.figure(figsize=(20, 15))  # Dimensioni più grandi per la leggibilità
    sns.heatmap(cm, annot=False, fmt="d", cmap="viridis", xticklabels=class_names, yticklabels=class_names)
    plt.title("Confusion Matrix Heatmap")
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.show()
    return cm

def evaluate_model_with_metrics(model, test_loader, class_names):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    all_labels = []
    all_predictions = []

    # Raccolta delle predizioni e dei label reali
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

    # Calcolo delle metriche
    report = classification_report(all_labels, all_predictions, target_names=class_names)
    
    # Stampa del classification report
    print("Classification Report:\n")
    print(report)
    
    return report


def load_model(model_path, num_classes):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Crea un'istanza del modello
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, num_classes)  # Adattamento delle classi
    model = model.to(device)
    
    # Carica i pesi salvati
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    return model

def save_confusion_matrix(cm, class_names, filename="confusion_matrix.png"):
    plt.figure(figsize=(20, 15))
    sns.heatmap(cm, annot=False, fmt="d", cmap="viridis", xticklabels=class_names, yticklabels=class_names)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.savefig(filename, dpi=300)
    plt.close()

def calculate_class_weights(dataset):
    # Conta il numero di campioni per ciascuna classe
    class_counts = [0] * len(dataset.classes)
    for _, label in dataset:
        class_counts[label] += 1
    
    # Calcolo dei pesi come inverso della frequenza delle classi
    total_samples = sum(class_counts)
    class_weights = [total_samples / count for count in class_counts]
    
    # Normalizzazione opzionale dei pesi
    class_weights = torch.tensor(class_weights, dtype=torch.float)
    return class_weights

# Main script
if __name__ == "__main__":

    train_loader, test_loader, class_names = preprocess_data()
    #model = train_model_resnet50(train_loader, test_loader, class_names) #allenamento del modello
    model = load_model("pokemon_classifier.pth", len(class_names)) #caricamento del modello per vedere i grafici e il class report
    cm = evaluate_model_with_confusion_matrix(model, test_loader, class_names)
    report = evaluate_model_with_metrics(model, test_loader, class_names)
    save_confusion_matrix(cm, class_names)
    