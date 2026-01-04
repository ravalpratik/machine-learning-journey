import pandas as pd

data = {
    "Name" : ["parth","pratik","bhavy","ankit","harsh"],
    "Age" : [23,None,44,23,None],
    "Salary" : [50000, 60000, 70000, None, None]
}
# Create a dictionary named 'data' to store information about individuals.
# Each key represents a column name, and its value is a list of data for that column.

df = pd.DataFrame(data)
print("Original DataFrame: ")
print(df)

print(df.isnull().sum())

df_drop = df.dropna()
print(df_drop)

df["Age"].fillna(df["Age"].mean(), inplace=True)
df["Salary"].fillna(df["Salary"].mean(), inplace=True)
print(df)

print(df.isnull().mean() * 100)