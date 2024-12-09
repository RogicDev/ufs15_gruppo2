import tkinter as tk
from tkinter import filedialog, Label, Button
from PIL import Image, ImageTk
import torch
from torchvision import transforms
import pandas as pd
from torchvision import models
import os
import pyttsx3  # Libreria per Text-to-Speech
import pygame  # Per la musica e i suoni

# Percorsi dei file globali
MODEL_PATH = "pokemon_classifier.pth"
POKEDEX_PATH = "data_pokedex/pokedex.csv"
DATASET_DIR = "data"
BACKGROUND_MUSIC_PATH = "background_music.mp3"
BUTTON_SOUND_PATH = "button_click.wav"

# Inizializza pygame per la musica e i suoni
pygame.mixer.init()
pygame.mixer.music.load(BACKGROUND_MUSIC_PATH)  # Carica la musica di background
pygame.mixer.music.play(-1)  # Riproduci in loop la musica di background
pygame.mixer.music.set_volume(0.1)

button_click_sound = pygame.mixer.Sound(BUTTON_SOUND_PATH)  # Carica il suono del click
button_click_sound.set_volume(0.5)

# Caricamento del modello
def load_model(class_names):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.resnet50(pretrained=True)
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
    #Prima: [channels, height, width] (3, 128, 128)
    return transform(image).unsqueeze(0)
    #Dopo: [batch_size, channels, height, width] (1, 3, 128, 128)
    #Batch size necessario perché rappresenta il numero di esempi che vengono processati dal modello durante un'iterazione

# Predizione con l'uso della funzione pokedex
def pokedex_info(prediction, pokedex):
    row = pokedex[pokedex['Nome'].str.strip().str.lower() == prediction.strip().lower()]
    if not row.empty:
        description, tipo = row['Descrizione'].iloc[0], row['Tipo'].iloc[0]
        return prediction, tipo, description
    else:
        return prediction, "Sconosciuto", "Descrizione non disponibile"

# Nuova funzione predict che usa pokedex_info
def predict(image_path, model, class_names, pokedex):
    image_tensor = preprocess_image(image_path).to(model[1])
    with torch.no_grad():
        outputs = model[0](image_tensor)
        _, predicted = torch.max(outputs, 1)
        pokemon_name = class_names[predicted.item()]
        pokemon_name, pokemon_type, pokemon_desc = pokedex_info(pokemon_name, pokedex)
        return pokemon_name, pokemon_type, pokemon_desc

# Funzione per Text-to-Speech
def speak(text):
    engine = pyttsx3.init()
    voices = engine.getProperty('voices')
    for voice in voices:
        if "italian" in voice.languages:
            engine.setProperty('voice', voice.id)
            break
    engine.say(text)
    engine.runAndWait()

# Funzione per caricare immagine
def open_file():
    button_click_sound.play()  # Suona il suono del click
    filepath = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg *.png *.jpeg, *.webp")])
    if filepath:
        img = Image.open(filepath)
        img.thumbnail((300, 300))
        img = ImageTk.PhotoImage(img)
        image_label.configure(image=img)
        image_label.image = img
        image_label.filepath = filepath

# Funzione per fare la predizione e aggiornare i risultati
def classify_image():
    # button_click_sound.play()  # Suona il suono del click
    filepath = getattr(image_label, 'filepath', None)
    if not filepath:
        result_label.config(text="Nessuna immagine selezionata!")
        return
    pokemon_name, pokemon_type, pokemon_desc = predict(filepath, model, class_names, pokedex)
    result_label.config(text=f"Nome: {pokemon_name}\nTipo: {pokemon_type}\nDescrizione: {pokemon_desc}")
    audio_text = f"Il Nome è {pokemon_name}. Il tipo è {pokemon_type}. {pokemon_desc}."
    result_label.update_idletasks()
    speak(audio_text)

#---------------------------------------------------------------------------------------------------------

# GUI
app = tk.Tk()
app.title("Pokédex")
app.geometry("400x600")

# Titolo in alto
title_label = Label(app, text="Who's that Pokémon?", font=("Helvetica", 24, "bold"), fg="red")
title_label.pack(pady=10)

# Immagine caricata
image_label = Label(app)
image_label.pack(pady=10)

# Bottone per selezionare immagine
open_button = Button(
    app,
    text="Seleziona Immagine",
    command=open_file,
    font=("Helvetica", 12, "bold"),
    bg="blue",
    fg="white",
    activebackground="darkblue",
    activeforeground="white"
)
open_button.pack(pady=10)

# Bottone per classificare immagine
classify_button = Button(
    app,
    text="Classifica Immagine",
    command=classify_image,
    font=("Helvetica", 12, "bold"),
    bg="green",
    fg="white",
    activebackground="darkgreen",
    activeforeground="white"
)
classify_button.pack(pady=10)

# Risultati della classificazione
result_label = Label(
    app,
    text="",
    font=("Helvetica", 12),
    wraplength=350,
    justify="left",
    bg="lightyellow",
    fg="black"
)
result_label.pack(pady=10)

# Caricamento modello e Pokedex
pokedex = pd.read_csv(POKEDEX_PATH)
class_names = os.listdir(DATASET_DIR)
model = load_model(class_names)

app.mainloop()