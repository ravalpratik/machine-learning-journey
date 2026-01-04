import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error,r2_score
import numpy as np

df = pd.read_csv("student.csv")
print(df)

x = df[["Hours"]]
y = df["Score"]

model = LinearRegression()
model.fit(x,y)

predicted_score = model.predict(x)
print(predicted_score)

mae = mean_absolute_error(y,predicted_score)
mse = mean_squared_error(y,predicted_score)
rmse = np.sqrt(mse)

print(f"MAE Mean Absolute Error {mae}")
print(f"MSE Mean Squared Error {mse}")
print(f"Root Mean Squared Error {rmse}")

new_predict = model.predict([[7]])
print(new_predict)