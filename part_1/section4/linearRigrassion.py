from sklearn.linear_model import LinearRegression

x = [[1],[2],[3],[4],[5]]
y = [40,50,65,75,90]

model = LinearRegression()

model.fit(x,y)
hours = float(input("Enter the hours you studied: "))

predicted_marks = model.predict([[hours]])
print(f"Based on your study hours {hours}, your predicted marks is {predicted_marks}")