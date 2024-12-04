import pandas as pd

def pokedex(prediction):
    df = pd.read_csv(rf'data/pokedex.csv')
    row = df[df['Nome'].str.strip().str.lower() == prediction.strip().lower()]
    description = row['Descrizione'].iloc[0]
    return description

if __name__ == '__main__':
    description = pokedex('Venusaur')
    print(description)