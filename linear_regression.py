from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from random import randint 
from sklearn.metrics import accuracy_score

X = [(randint(0, 1), randint(0, 1)) for _ in range(500)]
y = [(a^b) for (a, b) in X]
X = [(a, b, a|b)) for (a, b) in X] # add third dimension to allow linear model to find a seperator of the space

model = LinearRegression()
# model = SVR()
model.fit(X[:100], y[:100])
preds = [round(x) for x in model.predict(X[100:])]
accuracy_score(preds, y[100:])

plt.show()
# plt.plot([-1, 30])
accuracy_score(preds, y[100:]), model.coef_
