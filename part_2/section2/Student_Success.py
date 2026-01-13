import pandas as pd
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv("Student_data.csv")

# sample rows
print(df.head())

# dataset shape
print(f"rows: {df.shape[0]} , columns: {df.shape[1]}")

# transforming categorical columns using label encoding
le = LabelEncoder()
df["Internet"] = le.fit_transform(df["Internet"])   # yes = 1 , no = 0
df["Passed"] = le.fit_transform(df["Passed"])   # yes = 1 , no = 0

# dataset information
print(df.info())

# summary statistics
print(df.describe(include="all"))

# Missing values
print(df.isnull().sum())

