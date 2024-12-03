import tkinter as tk
from tkinter import filedialog, Label, Button
from PIL import Image, ImageTk
import torch
from torchvision import transforms
import pandas as pd
from torchvision import models
import os
import pyttsx3  # Libreria per Text-to-Speech

# Percorsi dei file
MODEL_PATH = "pokemon_classifier.pth"
POKEDEX_PATH = "Pokedex.xlsx"
DATASET_DIR = "data"

# Caricamento del modello
def load_model(class_names):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.resnet18(pretrained=True)
    model.fc = torch.nn.Linear(model.fc.in_features, len(class_names))
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model = model.to(device)
    model.eval()
    return model, device

# Preprocessing immagine
def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    image = Image.open(image_path).convert("RGB")
    return transform(image).unsqueeze(0)

# Predizione
def predict(image_path, model, class_names, pokedex):
    image_tensor = preprocess_image(image_path).to(model[1])
    with torch.no_grad():
        outputs = model[0](image_tensor)
        _, predicted = torch.max(outputs, 1)
        pokemon_name = class_names[predicted.item()]
        pokemon_info = pokedex.loc[pokedex['Nome'] == pokemon_name].iloc[0]
        return pokemon_name, pokemon_info['Tipo'], pokemon_info['Descrizione']

# Funzione per Text-to-Speech
def speak(text):
    engine = pyttsx3.init()
    # Configura la voce italiana
    voices = engine.getProperty('voices')
    for voice in voices:
        if "italian" in voice.languages:  # Cerca una voce italiana
            engine.setProperty('voice', voice.id)
            break
    # Leggi il testo
    engine.say(text)
    engine.runAndWait()

# Funzione per caricare immagine
def open_file():
    filepath = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg *.png *.jpeg")])
    if filepath:
        img = Image.open(filepath)
        img.thumbnail((300, 300))
        img = ImageTk.PhotoImage(img)
        image_label.configure(image=img)
        image_label.image = img
        image_label.filepath = filepath

# Funzione per fare la predizione e aggiornare i risultati
def classify_image():
    filepath = getattr(image_label, 'filepath', None)
    if not filepath:
        result_label.config(text="Nessuna immagine selezionata!")
        return
    # Predizione
    pokemon_name, pokemon_type, pokemon_desc = predict(filepath, model, class_names, pokedex)
    # Stampa del testo prima dell'audio
    result_label.config(text=f"Nome: {pokemon_name}\nTipo: {pokemon_type}\nDescrizione: {pokemon_desc}")
    # Genera audio
    audio_text = f"Il Nome è {pokemon_name}. Il tipo è {pokemon_type}. {pokemon_desc}."
    result_label.update_idletasks()  # Forza l'aggiornamento immediato del testo nella GUI
    speak(audio_text)

# GUI
app = tk.Tk()
app.title("Classificatore Pokémon")
app.geometry("400x600")

image_label = Label(app)
image_label.pack(pady=10)

open_button = Button(app, text="Seleziona immagine", command=open_file)
open_button.pack(pady=10)

classify_button = Button(app, text="Classifica immagine", command=classify_image)
classify_button.pack(pady=10)

result_label = Label(app, text="", wraplength=350, justify="left")
result_label.pack(pady=10)

# Caricamento modello e Pokedex
pokedex = pd.read_excel(POKEDEX_PATH)
class_names = os.listdir(DATASET_DIR)
model = load_model(class_names)

app.mainloop()
