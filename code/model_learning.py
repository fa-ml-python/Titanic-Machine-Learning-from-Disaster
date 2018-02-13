from pandas import read_csv
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

X_train = read_csv("../data/factors/x_train.csv")
Y_train = read_csv("../data/factors/y_train.csv")
X_test = read_csv("../data/factors/x_test.csv")


# Logistic Regression
logreg = LogisticRegression()
logreg.fit(X_train, Y_train)
Y_pred = logreg.predict(X_test)
print(logreg.score(X_train, Y_train))


def overfitting_curve(model, x, y,
                      averaging_factor: int = 50, plot: bool =False) -> tuple:
    """ Метод строит кривые обучения.

    Используется для диагностики моделей машиного обучения
    """

    m, n = x.shape
    assert m == len(y), "Несоответствующие по длине наборы данных"

    sizes_of_train_set = []
    train_errors = []
    test_errors = []
    batch_size = 20

    while batch_size <= m - 1:
        step_train_errors = []
        step_test_errors = []
        print("Overfitting curve step", batch_size, "of", m)
        for step in range(averaging_factor):
            x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=batch_size)
            model.fit(x_train, y_train)
            step_train_errors.append(1.0 - model.score(x_train, y_train))
            step_test_errors.append(1.0 - model.score(x_test, y_test))
        sizes_of_train_set.append(batch_size + 1)
        train_errors.append(sum(step_train_errors) / len(step_train_errors))
        test_errors.append(sum(step_test_errors) / len(step_test_errors))
        batch_size += max(int(m / 50), 1)

    if plot:
        import matplotlib.pyplot as plt
        plt.title('Overfitting curve')
        plt.plot(sizes_of_train_set, train_errors, color='blue', label='Train score')
        plt.plot(sizes_of_train_set, test_errors, color='red', label='Validation score')
        plt.ylim(ymin=0.0)
        plt.xlabel('Size of a fit set')
        plt.ylabel('Error')
        plt.legend()
        plt.show()

    return sizes_of_train_set, train_errors, test_errors


log_reg_model = LogisticRegression()
overfitting_curve(log_reg_model, X_train, Y_train, plot=True)


log_reg_model = SVC()
overfitting_curve(log_reg_model, X_train, Y_train, plot=True)


log_reg_model = RandomForestClassifier(n_estimators=100)
overfitting_curve(log_reg_model, X_train, Y_train, plot=True)
