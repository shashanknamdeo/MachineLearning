import pandas as pd

path = r'C:\Users\ShirishPC\Downloads\titanic\train.csv'

df = pd.read_csv(path)

df = df.drop(['Cabin'], axis=1)
df = df.dropna()
df = df.reset_index(drop=True)

