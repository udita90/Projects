import pandas as pd

from sklearn.datasets import fetch_california_housing

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import matplotlib.pyplot as plt

import numpy as np

data = fetch_california_housing(as_frame=True)

df = pd.concat([data.data, data.target.rename("MedHouseval")], axis=1)

df.head()


X = df.drop(columns='MedHouseval')

y = df["MedHouseval"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



model = LinearRegression()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)


mae = mean_absolute_error(y_test, y_pred)

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

r2 = r2_score(y_test, y_pred)


print(f"MAE: {mae:.3f} RMSE: {rmse:.3f} R2: {r2:.3f}")

plt.scatter(y_test, y_pred, alpha=0.4)

plt.xlabel("Actual")

plt.ylabel("Predicted")

plt.title("Actual vs Predicted")

plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red')

plt.show()
