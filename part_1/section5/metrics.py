from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# true answers (what actually happened)

y_true = [1,0,1,1,0,1,0]

# model predictions (what it predicted)

y_pred = [1,0,1,0,0,1,1]

# evaluate the model

print("Accuracy: ", accuracy_score(y_true,y_pred))
print("precision: ", precision_score(y_true,y_pred))
print("recall: ", recall_score(y_true,y_pred))
print("f1: ", f1_score(y_true,y_pred))