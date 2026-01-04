import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

# Load the dataset
df = pd.read_csv("student_data.csv")
print(df.head(10))

x = df[["Hours_Studied"]]
y = df["Exam_Score"]

# Create and train the model
model = LinearRegression()
model.fit(x,y)
model_score = model.predict(x)

# Calculate the metrics
mae = mean_absolute_error(y, model_score)
msr = mean_squared_error(y, model_score)
rmse = np.sqrt(msr)
r2 = r2_score(y, model_score)

print(f"Mean Absolute Error: {round(mae,2)}")
print(f"Mean Squared Error: {round(msr,2)}")
print(f"Root Mean Squared Error: {round(rmse,2)}")
print(f"r^2 score is: {round(r2,4)}")       #it's colser to 1 the better the model

plt.figure(figsize=(10,6))
plt.hist(df["Exam_Score"], bins=20, color="blue", edgecolor="black")
plt.title("Distribution of Exam Scores")
plt.xlabel("Exam Score")
plt.ylabel("number of Students")
plt.grid(True)
plt.show()

plt.figure(figsize=(10,6))
plt.scatter(x, y, color="blue",label="Actual Scores")
plt.plot(x, model_score, color="red", label="Predicted Scores")
plt.title("model Prediction vs Actual Scores")
plt.xlabel("Exam Score")
plt.ylabel("number of Students")
plt.grid(True)
plt.show()