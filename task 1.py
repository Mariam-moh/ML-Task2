from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt


X = np.array([[3], [4], [5], [6], [7]])

y = np.array([50, 60, 65, 75, 85])


model = LinearRegression()
model.fit(X, y)

X_new = np.array([[8]])
y_pred = model.predict(X_new)
print(f"Predicted grade for 8 hours of sleep: {y_pred[0]:.2f}")


plt.scatter(X, y, color='blue', label='Data')
plt.plot(X, model.predict(X), color='red', label='Fit Line')
plt.xlabel('Sleep Hours')
plt.ylabel('Exam Grade')
plt.title('Linear Regression Example')
plt.legend()
plt.show()
