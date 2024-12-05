import pandas as pd

def pokedex(prediction):
    df = pd.read_csv(rf'ufs15_gruppo2/data_pokedex/pokedex.csv')
    row = df[df['Nome'].str.strip().str.lower() == prediction.strip().lower()]
    description, tipo = row['Descrizione'].iloc[0], row['Tipo'].iloc[0]
    return prediction, description, tipo

if __name__ == '__main__':
    nome, description, tipo = pokedex('Venusaur')
    print(f'Nome: {nome}\nDescrizione: {description}\nTipo: {tipo}')
