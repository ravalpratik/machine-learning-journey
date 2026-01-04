import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split

df = pd.read_csv("simple_data.csv")
print("Original DataFrame:")
print(df.head(5))

df_label = df.copy()

df_label = df_label.drop(columns=["Name","City"])
print(df_label.head(5))

# label endoding
le = LabelEncoder()
df_label["Gender_Encoded"] = le.fit_transform(df_label["Gender"])
df_label["Passed_Encoded"] = le.fit_transform(df_label["Passed"])
print("label encoded DataFrame: ")
print(df_label.head(5))

# deleting gender and passed columns from label encoded DataFrame
df_label = df_label.drop(columns=["Gender","Passed"])
print(df_label.head(5))

# standerd scaler
Standard_sclaler = StandardScaler()
Standard_sclaled = Standard_sclaler.fit_transform(df_label)
print("Standerd scaled DataFrame: ")
print(pd.DataFrame(Standard_sclaled, columns=["Gender","Passed"]).head(5))

# min max scaler
minmax_scaler = MinMaxScaler()
minmax_scalerd = minmax_scaler.fit_transform(df_label)
print("min max scaled DataFrame: ")
print(pd.DataFrame(minmax_scalerd, columns=["Gender","Passed"]).head(5))

# train test split
x = df_label[["Gender_Encoded"]]
y = df_label[["Passed_Encoded"]]

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=42)
print("Train Data: ")
print(x_train.head(5))
print("Test Data: ")
print(x_test.head(5))

print("Train labels: ")
print(y_train.head(5))

print("Test labels: ")
print(y_test.head(5))