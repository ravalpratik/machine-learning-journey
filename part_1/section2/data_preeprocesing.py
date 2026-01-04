from sklearn.preprocessing import LabelEncoder
import pandas as pd

df = pd.read_csv("simple_data.csv")

print("Original DataFrame:")
print(df.head())

df_label = df.copy()

le = LabelEncoder()

df_label["Gender_Encoded"] = le.fit_transform(df_label["Gender"])
df_label["Passed_Encoded"] = le.fit_transform(df_label["Passed"])


print("\nlabel encoded DataFrame: ")
print(df_label[["Gender_Encoded", "Gender","Passed_Encoded","Passed"]].head())

print("\none-hot encoded DataFrame: ")
df_incoded = pd.get_dummies(df_label, columns = ["City"])
print(df_incoded.head())