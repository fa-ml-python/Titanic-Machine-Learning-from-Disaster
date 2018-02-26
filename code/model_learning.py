import warnings

from pandas import read_csv, concat
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings('ignore')

X = read_csv("../data/factors/x_train.csv")
Y = read_csv("../data/factors/y_train.csv")
data = concat([Y, X], axis=1)

train, test = train_test_split(data, test_size=0.3, random_state=0, stratify=data['Survived'])
X_train = train[train.columns[1:]]
Y_train = train[train.columns[:1]]
X_test = test[test.columns[1:]]
Y_test = test[test.columns[:1]]

models = (('Log regression', LogisticRegression()),
          ('Radial SVM', SVC(kernel='rbf', C=1, gamma=0.1)),
          ('Linear SVM', SVC(kernel='linear', C=1, gamma=0.1)),
          ('Decision tree', DecisionTreeClassifier()),
          ('K nearest', KNeighborsClassifier(n_neighbors=5)),
          ('Naive Bayesian', GaussianNB()),
          ('Random forest', RandomForestClassifier(n_estimators=100)),
          ('Perceptron', MLPClassifier(solver='lbfgs', hidden_layer_sizes=(18,))),
          ('Wide net', MLPClassifier(solver='lbfgs', hidden_layer_sizes=(90,))),
          ('Deep net', MLPClassifier(solver='lbfgs', hidden_layer_sizes=(18, 18, 18, 18))),
          ('Voting', VotingClassifier(estimators=[('KNN', KNeighborsClassifier(n_neighbors=10)),
                                                  ('RBF', SVC(probability=True, kernel='rbf', C=0.5, gamma=0.1)),
                                                  ('RFor', RandomForestClassifier(n_estimators=500, random_state=0)),
                                                  ('LR', LogisticRegression(C=0.05)),
                                                  ('DT', DecisionTreeClassifier(random_state=0)),
                                                  ('NB', GaussianNB()),
                                                  ('svm', SVC(kernel='linear', probability=True))
                                                  ],
                                      voting='soft').fit(X_train, Y_train)),
          ('AdaBoost', AdaBoostClassifier(n_estimators=200, random_state=0, learning_rate=0.1)),
          ('GradientBoost', GradientBoostingClassifier(n_estimators=500, random_state=0, learning_rate=0.1)),
          )

from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
import pandas as pd

kfold = KFold(n_splits=20)

accuracy = []
std = []
for name, model in models:
    cv_result = cross_val_score(model, X, Y, cv=kfold, scoring="accuracy")
    cv_result = cv_result
    accuracy.append(cv_result.mean())
    std.append(cv_result.std())
new_models_dataframe2 = pd.DataFrame({'CV Mean': accuracy, 'Std': std}, index=[name for name, model in models])
print(new_models_dataframe2)


print()


# from sklearn.model_selection import GridSearchCV
#
# C=[0.05,0.1,0.2,0.3,0.25,0.4,0.5,0.6,0.7,0.8,0.9,1]
# kernel=['rbf','linear']
# hyper={'kernel':kernel,'C':C}
# gd=GridSearchCV(estimator=SVC(),param_grid=hyper,verbose=True)
# gd.fit(X,Y)
# print("Best SVM model", gd.best_score_)
# print(gd.best_estimator_)
#
# C=[0.05,0.1,0.2,0.3,0.25,0.4,0.5,0.6,0.7,0.8,0.9,1]
# solver=['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']
# hyper={'C':C,'solver':solver}
# gd=GridSearchCV(estimator=LogisticRegression(),param_grid=hyper,verbose=True)
# gd.fit(X,Y)
# print("Best linear regression model", gd.best_score_)
# print(gd.best_estimator_)
#
# n_estimators=list(range(100,1100,100))
# learn_rate=[0.05,0.1,0.2,0.3,0.25,0.4,0.5,0.6,0.7,0.8,0.9,1]
# hyper={'n_estimators':n_estimators,'learning_rate':learn_rate}
# gd=GridSearchCV(estimator=AdaBoostClassifier(),param_grid=hyper,verbose=True)
# gd.fit(X,Y)
# print("Best AdaBoost model", gd.best_score_)
# print(gd.best_estimator_)
#
# n_estimators=list(range(100,1100,100))
# learn_rate=[0.05,0.1,0.2,0.3,0.25,0.4,0.5,0.6,0.7,0.8,0.9,1]
# hyper={'n_estimators':n_estimators,'learning_rate':learn_rate}
# gd=GridSearchCV(estimator=GradientBoostingClassifier(),param_grid=hyper,verbose=True)
# gd.fit(X,Y)
# print("Best GradientBoost model", gd.best_score_)
# print(gd.best_estimator_)


# from sklearn.ensemble import BaggingClassifier
# model=BaggingClassifier(base_estimator=LogisticRegression(),random_state=0,n_estimators=700)
# model.fit(X_train,Y_train)
# prediction=model.predict(X_test)
# print('The accuracy for bagged LG is:',metrics.accuracy_score(prediction,Y_test))
# result=cross_val_score(model,X,Y,cv=10,scoring='accuracy')
# print('The cross validated score for bagged LR is:',result.mean())
#
# model=BaggingClassifier(base_estimator=SVC(kernel='linear'))
# model.fit(X_train,Y_train)
# prediction=model.predict(X_test)
# print('The accuracy for bagged SVC is:',metrics.accuracy_score(prediction,Y_test))
# result=cross_val_score(model,X,Y,cv=10,scoring='accuracy')
# print('The cross validated score for bagged SVC is:',result.mean())
