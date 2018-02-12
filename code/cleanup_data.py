""" Начинаем исследовать набор данных.

Нам понадобятся модули pandas
"""
from numpy import random, isnan
from pandas import read_csv

"""
Импорт данных
"""
titanic_df = read_csv("D:\\GitHub\\Titanic-Machine-Learning-from-Disaster\\data\\raw_data\\train.csv")
test_df = read_csv("D:\\GitHub\\Titanic-Machine-Learning-from-Disaster\\data\\raw_data\\test.csv")

"""
Для начала, удалим ненужные нам колонки: 'PassengerId','Name','Ticket'.
Это поля, уникальные для каждого пассажира, и, поэтому, не несущие информации о взаимосвязях в данных.
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
"""

"""
В поле 'Embarked' отсутствуют всего два значения только в обучающей выборке. В тестовой - все значения в наличии.
Поэтому, подставим в эти две точки наиболее распространенное значение (а именно, 'S')
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
average_age_titanic = titanic_df["Age"].mean()      # среднее в выборке
std_age_titanic = titanic_df["Age"].std()       # вариация выборки
count_nan_age_titanic = titanic_df["Age"].isnull().sum()        # количество отсутствующих значений
# то же самое по тестовой выборке
average_age_test = test_df["Age"].mean()
std_age_test = test_df["Age"].std()
count_nan_age_test = test_df["Age"].isnull().sum()
# генерируем набор случайных числе в диапазоне +- одна вариация от среднего
rand_1 = random.randint(average_age_titanic - std_age_titanic,
                        average_age_titanic + std_age_titanic,
                        size=count_nan_age_titanic)
rand_2 = random.randint(average_age_test - std_age_test,
                        average_age_test + std_age_test,
                        size=count_nan_age_test)
# заполняем отсутствующие точки данных сгенерированными значениями
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
Мы закончили заполнять отсутствующие данные. Запишем таблицы дла последующего дискриптивного анализа
"""
titanic_df.to_csv('../data/tidy_data/train.csv', index=False)
test_df.to_csv('../data/tidy_data/test.csv', index=False)

