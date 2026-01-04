"""MEAN ABSOLUTE ERROR
1- take the mistake diffrence
2- remove the negative sign
3- add
4- divide by the number of samples

7.5 on avrage"""

"""MSE MEAN SQUARED ERROR
1- mistake square them
2- add
3- divide total

MSE = 62.5"""

"""RMSE ROOT MEAN SQUARED ERROR
"""

from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

# real scores
real_scores = [90,60,80,100]

# machine predicted scores
pred_scores = [85,70,70,95]

mae = mean_absolute_error(real_scores, pred_scores)

mse = mean_squared_error(real_scores, pred_scores)

rmse = np.sqrt(mse)

print(f"MAE: on avrage by {mae}")
print(f"MSE: squared mistake value {mse}")
print(f"RMSE: final realistic error {rmse}")