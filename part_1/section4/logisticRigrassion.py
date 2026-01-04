from sklearn.linear_model import LogisticRegression

x = [[1],[2],[3],[4],[5]]       #studeys hours
y = [0,0,1,1,1]                 #results (0 for failed, 1 for passed)

model = LogisticRegression()
model.fit(x,y)

hours = float(input("Enter the number of hours studied: "))

predicted_vallue = model.predict([[hours]])[0]

if predicted_vallue == 1:
    print(f"Based on the input of {hours} hours stady, the student will pass.")
else:
    print(f"Based on the input of {hours} hours study, the student will fail.")