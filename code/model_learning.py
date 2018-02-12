from pandas import read_csv
from sklearn.linear_model import LogisticRegression

X_train = read_csv("../data/factors/x_train.csv")
Y_train = read_csv("../data/factors/y_train.csv")
X_test = read_csv("../data/factors/x_test.csv")

# Logistic Regression

logreg = LogisticRegression()
logreg.fit(X_train, Y_train)
Y_pred = logreg.predict(X_test)
logreg.score(X_train, Y_train)