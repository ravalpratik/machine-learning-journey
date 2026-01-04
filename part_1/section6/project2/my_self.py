import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error,r2_score
from sklearn.preprocessing import LabelEncoder

#createing a dataframe
df = pd.read_csv("student_data.csv")

# finding the null values
# print("printing thr null values")
# print(df.isnull().sum())
# df.dropna(inplace=True)
# print(df.isnull().sum())

#making a copy of the dataframe
df_copy = df.copy()
print(df_copy.head())


# df_copy.info()

#deleteing the coulumns which are not required
df_copy.drop(columns=["Parental_Involvement","School_Type","Parental_Education_Level","Distance_from_Home","Gender"],inplace=True)
print(df_copy.head())


#encoding categaorical data

le = LabelEncoder()

# changing the values of Access_to_resources columns
df_copy["Access_to_Resources"] = le.fit_transform(df_copy["Access_to_Resources"])
print(df_copy["Access_to_Resources"].head())
# 0 for High 1 for low 2 for Medium

df_copy["Extracurricular_Activities"] = le.fit_transform(df_copy["Extracurricular_Activities"])
print(df_copy["Extracurricular_Activities"].head())
# 0 for no 1 for yes

df_copy["Motivation_Level"] = le.fit_transform(df_copy["Motivation_Level"])
print(df_copy["Motivation_Level"].head())
# 0 for high 1 for low 2 for medium

df_copy["Internet_Access"] = le.fit_transform(df_copy["Internet_Access"])
print(df_copy["Internet_Access"].head())
# 0 for no and 1 for yes

df_copy["Family_Income"] = le.fit_transform(df_copy["Family_Income"])
print(df_copy["Family_Income"].head())
# 0 for high 1 for low 2 for medium

df_copy["Teacher_Quality"] = le.fit_transform(df_copy["Teacher_Quality"])
print(df_copy["Teacher_Quality"].head())
# 0 for high 1 for low 2 for medium

df_copy["Peer_Influence"] = le.fit_transform(df_copy["Peer_Influence"])
print(df_copy["Peer_Influence"].head())
# 0 for negative 1 for netural 2 for positive

df_copy["Learning_Disabilities"] = le.fit_transform(df_copy["Learning_Disabilities"])
print(df_copy["Learning_Disabilities"].head())
# 0 for no and 1 for yes

# difining the x and y avlues
x = df_copy[["Hours_Studied","Attendance","Access_to_Resources","Extracurricular_Activities","Sleep_Hours","Previous_Scores","Motivation_Level","Internet_Access","Tutoring_Sessions","Family_Income","Teacher_Quality","Peer_Influence","Physical_Activity","Learning_Disabilities"]]
y = df_copy["Exam_Score"]

# splitting the data into training and tasting data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)


# creating the model
model = LinearRegression()
model.fit(x_train, y_train)

# evaluating the model
y_pred = model.predict(x_test)


test_fiture = pd.DataFrame(x_test)
test_label = pd.DataFrame(x_test)
predicted_label = pd.DataFrame(y_pred)

# metrics
mae = mean_absolute_error(y_test, y_pred)
print("mean absolute error:", round(mae, 4))

mse = mean_squared_error(y_test, y_pred)
print("mean squared error: ", round(mse, 4))

rmse = np.sqrt(mse)
print("root mean squared error: ", round(rmse, 4))

r2 = r2_score(y_test,y_pred)
print("r2 score: ", round(r2, 4))

# visualization
plt.scatter(test_fiture, test_label, label="Actual_data")
plt.plot(test_fiture, y_pred, label="predicted_data")
plt.xlabel("Hours_Studied")
plt.ylabel("Exam_Score")
# plt.legend()
plt.show()
# df_copy.info()
