import pandas as pd
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.model_selection import train_test_split

data = {
    "StudyHours" : [1,2,3,4,5],
    "TestScores" : [40,50,60,70,80]
}

df = pd.DataFrame(data)

# standerd Scaler

standerd_scaler = StandardScaler()
standerd_scaled = standerd_scaler.fit_transform(df)

print("Standerd Scaled DataFrame: ")
print(pd.DataFrame(standerd_scaled, columns=["StudyHours", "TestScores"]))

# min max scaler
minmax_scaler = MinMaxScaler()
minmax_scaled = minmax_scaler.fit_transform(df)

print("\nminmax scaled")
minmax = pd.DataFrame(minmax_scaled, columns=["StydyHours","TestScores"])
print(minmax)
print(pd.DataFrame(minmax_scaled, columns=["StudyHours","TestScore"]))

# train test split
x = df[["StudyHours"]]
y = df[["TestScores"]]

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2, random_state=42)

print("Train Data: ")
print(x_train)

print("test Dtata: ")
print(x_test)

print("Train labels:")
print(y_train)

print("test labels:")
print(y_test)   