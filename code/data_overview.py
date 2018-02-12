""" Начинаем исследовать набор данных.

Нам понадобятся модули pandas
"""
from numpy import random, isnan
from pandas import read_csv, get_dummies
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

"""
Импорт данных
"""
titanic_df = read_csv("D:\\GitHub\\Titanic-Machine-Learning-from-Disaster\\data\\raw_data\\train.csv")
test_df = read_csv("D:\\GitHub\\Titanic-Machine-Learning-from-Disaster\\data\\raw_data\\test.csv")

"""
Для начала, удалим ненужные нам колонки: 'PassengerId','Name','Ticket'.
Это поля, уникальные для каждого пассажира и, поэтому, не несущие информации о взаимосвязях в данных.
"""
titanic_df = titanic_df.drop(['PassengerId', 'Name', 'Ticket'], axis=1)
test_df = test_df.drop(['PassengerId', 'Name', 'Ticket'], axis=1)

"""
Теперь начнем разбираться с отсутствующими данными. 

Для анализа отсутствующих значений воспользуемся функцией info модуля pandas:
>>> titanic_df.info()
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 891 entries, 0 to 890
Data columns (total 9 columns):
Survived    891 non-null int64
Pclass      891 non-null int64
Sex         891 non-null object
Age         714 non-null float64
SibSp       891 non-null int64
Parch       891 non-null int64
Fare        891 non-null float64
Cabin       204 non-null object
Embarked    889 non-null object
dtypes: float64(2), int64(4), object(3)
memory usage: 62.7+ KB

>>> test_df.info()
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 418 entries, 0 to 417
Data columns (total 8 columns):
Pclass      418 non-null int64
Sex         418 non-null object
Age         332 non-null float64
SibSp       418 non-null int64
Parch       418 non-null int64
Fare        417 non-null float64
Cabin       91 non-null object
Embarked    418 non-null object
dtypes: float64(2), int64(3), object(3)
memory usage: 26.2+ KB

Мы видим, что есть четыре проблемные колонки: 'Age', 'Fare', 'Cabin' и 'Embarked'. Причем, ситуация у них весьма разная.

Существует несколько способов как подчистить отсутствующие точки данных:
1. Удалить столбец полностью. Это довольно радикальный метод, который стоит применять только тогда, когда вы уверены,
что удаление всего столбца не приведет к значимой потере информации для анализа. Чаще всего этот метод применяют если
отсутствующих данных очень много, больше, чем имеющихся, а сам удаляемый элемент данных не представляет особой 
ценности для анализа.
2. Заполнить отсутствующие точки наиболее ожидаемым значением.
3. Использовать случайные значения с заданным распределением
4. Использовать рекомендательную систему.

В поле 'Embarked' отсутствуют всего два значения только в обучающей выборке. В тестовой - все значения в наличии.
Поэтому, подставим в оти две точки наиболее распространенное значение (а именно, 'S')
"""
titanic_df["Embarked"] = titanic_df["Embarked"].fillna("S")

"""
В поле 'Fare' также отсутствует всего одно значение (в тестовом наборе). Его тоже просто заполнить средним значением.
В данном случае, 
"""
test_df["Fare"].fillna(test_df["Fare"].median(), inplace=True)

"""
В поле 'Age' довольно много отсутствующих значений как в тестовой, так и в обучающей выборке. Однако, это поле является
численным, что сильно упрощает анализ. Мы можем выяснить параметры распределения этой переменной и воспольтзоваться ими 
для генерации ряда случайных значений, похожих на истинные 
"""
#
average_age_titanic = titanic_df["Age"].mean()
std_age_titanic = titanic_df["Age"].std()
count_nan_age_titanic = titanic_df["Age"].isnull().sum()
# get average, std, and number of NaN values in test_df
average_age_test = test_df["Age"].mean()
std_age_test = test_df["Age"].std()
count_nan_age_test = test_df["Age"].isnull().sum()
# generate random numbers between (mean - std) & (mean + std)
rand_1 = random.randint(average_age_titanic - std_age_titanic,
                        average_age_titanic + std_age_titanic,
                        size=count_nan_age_titanic)
rand_2 = random.randint(average_age_test - std_age_test,
                        average_age_test + std_age_test,
                        size=count_nan_age_test)
# fill NaN values in Age column with random values generated
titanic_df["Age"][isnan(titanic_df["Age"])] = rand_1
test_df["Age"][isnan(test_df["Age"])] = rand_2

"""
Поле 'Cabin' содержит очень много отсутствующих точек, гораздо больше, чем известных. Кроме этого, есть предположение,
что номер каюты имеет малое влияние на то, выживет пассажир или нет (в условиях известности других факторов).
Поэтому, удалим эту колонки из тестового и обучеющего наборов данных.
"""
titanic_df.drop("Cabin", axis=1, inplace=True)
test_df.drop("Cabin", axis=1, inplace=True)

"""
Поле 'Embarked' является категориальным с тремя значениями - S, Q, C. Заменим его фиктивной переменной (dummy 
variable), состоящей из трех колонок
"""
# создание фиктивных переменных
embark_dummies_titanic = get_dummies(titanic_df['Embarked'])
embark_dummies_test = get_dummies(test_df['Embarked'])
# включение их в массивы данных
titanic_df = titanic_df.join(embark_dummies_titanic)
test_df = test_df.join(embark_dummies_test)
# удаление исходных колонок
titanic_df.drop(['Embarked'], axis=1, inplace=True)
test_df.drop(['Embarked'], axis=1, inplace=True)

# """
# Создание суррогатного фактора 'Family'.
# Вместо двух колонок, хранящих количество братьев и родителей, создадим одну - количество на борту родственников.
# Можно сделать эту переменную бинарной, но мы теряем информацию и с количеством лучше точность предсказаний.
# """
# # Создание фиктивной переменной как суммы двух столбцов
# titanic_df['Family'] = titanic_df["Parch"] + titanic_df["SibSp"]
# # titanic_df['Family'].loc[titanic_df['Family'] > 0] = 1
# # titanic_df['Family'].loc[titanic_df['Family'] == 0] = 0
# test_df['Family'] = test_df["Parch"] + test_df["SibSp"]
# # test_df['Family'].loc[test_df['Family'] > 0] = 1
# # test_df['Family'].loc[test_df['Family'] == 0] = 0
# # удаление старых колонок
# # titanic_df = titanic_df.drop(['SibSp', 'Parch'], axis=1)
# # test_df = test_df.drop(['SibSp', 'Parch'], axis=1)

"""
Создание суррогатного фактора 'Person'.
Классифицируем пассажиров как 'Male', 'Female', 'Child'
"""
def get_person(passenger):
    age, sex = passenger
    return 'child' if age < 16 else sex
# Добавление суррогатного поля
titanic_df['Person'] = titanic_df[['Age', 'Sex']].apply(get_person, axis=1)
test_df['Person'] = test_df[['Age', 'Sex']].apply(get_person, axis=1)
# Удаление колонки 'Sex'
# titanic_df.drop(['Sex'], axis=1, inplace=True)
# test_df.drop(['Sex'], axis=1, inplace=True)

# create dummy variables for Person column, & drop Male as it has the lowest average of survived passengers
person_dummies_titanic = get_dummies(titanic_df['Person'])
# person_dummies_titanic.columns = ['Child', 'Female', 'Male']
person_dummies_test = get_dummies(test_df['Person'])
# person_dummies_test.columns = ['Child', 'Female', 'Male']
titanic_df = titanic_df.join(person_dummies_titanic)
test_df = test_df.join(person_dummies_test)
titanic_df.drop(['Person'], axis=1, inplace=True)
test_df.drop(['Person'], axis=1, inplace=True)

"""
Так же преобразуем категориальную переменную 'Pclass'
"""
# создание фиктивных переменных
pclass_dummies_titanic = get_dummies(titanic_df['Pclass'])
pclass_dummies_titanic.columns = ['Class_1', 'Class_2', 'Class_3']
pclass_dummies_test = get_dummies(test_df['Pclass'])
pclass_dummies_test.columns = ['Class_1', 'Class_2', 'Class_3']
titanic_df.drop(['Pclass'], axis=1, inplace=True)
test_df.drop(['Pclass'], axis=1, inplace=True)
titanic_df = titanic_df.join(pclass_dummies_titanic)
test_df = test_df.join(pclass_dummies_test)

print(titanic_df.head(15))


X_train = titanic_df.drop("Survived", axis=1)
Y_train = titanic_df["Survived"]
X_test = test_df.copy()

# Logistic Regression
logreg = LogisticRegression()
logreg.fit(X_train, Y_train)
print(logreg.score(X_train, Y_train))

# Random Forests
random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train, Y_train)
# Y_pred = random_forest.predict(X_test)
print(random_forest.score(X_train, Y_train))
