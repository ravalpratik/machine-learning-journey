import pandas as pd

df = pd.read_csv("salse_data.csv")
print(df)

print(df.isnull().sum())
print(df.isnull().mean()*100)

df_droped = df.dropna()
print(df_droped)

df["Quantity"].fillna(df["Quantity"].mean(), inplace=True)
df["Price"].fillna(df["Price"].mean(),inplace=True)
print(df)