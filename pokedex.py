import pandas as pd

def pokedex(prediction):
    df = pd.read_csv(rf'data/pokedex.csv')
    row = df[df['Nome'].str.strip().str.lower() == prediction.strip().lower()]
    description, tipo = row['Descrizione'].iloc[0], row['Tipo'].iloc[0]
    return description, tipo

if __name__ == '__main__':
    description, tipo = pokedex('Venusaur')
    print(f'Descrizione: {description}\nTipo: {tipo}')
