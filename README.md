# Progetto: Classificazione di Pokémon (1ª Generazione)

## Descrizione del progetto
Questo progetto utilizza modelli avanzati di Machine Learning per classificare immagini di Pokémon della prima generazione. L'obiettivo è creare un'applicazione desktop che consenta di caricare un'immagine di un Pokémon, elaborarla con un modello addestrato, e fornire predizioni sul Pokémon riconosciuto, insieme a una descrizione del Pokédex, il tipo e un fatto divertente.

- **Dataset utilizzato**: [Pokemon Gen 1 Dataset](https://www.kaggle.com/datasets/echometerhhwl/pokemon-gen-1-38914)
- **Passaggi principali**:
  1. Preprocessing delle immagini
  2. Allenamento del modello
  3. Testing del modello
  4. Creazione di un'applicazione desktop interattiva
  5. Aggiunta di dettagli ed extra per migliorare l'esperienza utente

---

## Dipendenze necessarie

Assicurati di avere installato Python 3.8 o versioni successive fino alla 3.12 (3.13 non supporta pytorch). Le librerie richieste per il progetto sono elencate di seguito:

### Librerie per il preprocessing, allenamento e testing:
- `os`: libreria standard per operazioni sui file e directory
- `pandas`: per gestire i dati in formato tabellare
- `torch`: framework principale per la creazione e addestramento di reti neurali
- `torchvision`: per modelli pre-addestrati e trasformazioni delle immagini
- `Pillow`: per gestire e manipolare immagini (richiede `ImageFile`)
- `scikit-learn`: per la suddivisione del dataset in training e test set (`train_test_split`)

### Librerie per l'interfaccia utente:
- `tkinter`: per creare interfacce grafiche (GUI)
- `pyttsx3`: per la sintesi vocale (Text-to-Speech)

### Altre librerie richieste:
- `Image` e `ImageTk` (da Pillow): per gestire immagini nell'applicazione GUI

---

## Installazione
`pip install torch torchvision pandas scikit-learn Pillow pyttsx3`
