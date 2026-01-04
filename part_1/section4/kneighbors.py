from sklearn.neighbors import KNeighborsClassifier

x = [
    [180,7],
    [200,7.5],
    [250,8],
    [300,8.5],
    [330,9],
    [360,9.5]
]

# 0 = Apple, 1 = Orange

y = [0,0,0,1,1,1]

model = KNeighborsClassifier(n_neighbors=3)

model.fit(x,y)

weight = float(input("enter weight in grams: "))
size = float(input("enter the size in CM: "))

prediction = model.predict([[weight,size]])[0]

if prediction == 0:
    print("this is likely an apple")
else:
    print("this likely an orange")