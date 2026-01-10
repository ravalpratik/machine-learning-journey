# PCA(principal component analysis) implementation using numpy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# sample data 
data = {
    "Age": [25,30,35,40,45,50],
    "Income" : [30000,40000,50000,60000,70000,80000],
    "Spending" : [70,60,50,40,30,20],
    "Savings" : [1000,5000,8000,10000,15000,20000]
}

df = pd.DataFrame(data)

scaler = StandardScaler()
scaled_data = scaler.fit_transform(df)

pca = PCA(n_components=2)
pca_result = pca.fit_transform(scaled_data)

pca_data = pd.DataFrame(pca_result, columns=["PCA1","PCA2"])

explained_variance= pca.explained_variance_ratio_
print("Variance captured by each PCA Component: ")
print(np.round(explained_variance * 100 ,2)) # %

plt.figure(figsize=(8,6))
plt.scatter(pca_data["PCA1"],pca_data["PCA2"],color="black",s=80)
plt.title("PCA projection (2d view)")
plt.xlabel("PCA 1 Main pattern")
plt.ylabel("PCA 2 minor pattern")
plt.grid(True)
plt.show()

print("New data with 2 features PCA1 PCA2")
print(pca_data)