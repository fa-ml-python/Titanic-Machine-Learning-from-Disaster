from numpy import isnan
from pandas import read_csv
import seaborn as sns
import matplotlib.pyplot as plt


# Импорт данных
data = read_csv("../data/tidy_data/train.csv")

"""
Ручное описание колонок даных:

Целевое поле:
    Survived
    
Поля-идентификаторы:
    PassengerId
    Name
    Ticket
    
Категориальные данные:
    Pclass
    Sex
    Cabin
    Embarked

Количественные данные:
    Age
    SibSp
    Parch
    Fare
    
"""

numerical_fields = ['Age', 'SibSp', 'Parch', 'Fare']

# for field in numerical_fields:
#     sns.distplot(data[field][~isnan(data[field])])
#     plt.show()


# sns.pairplot(data)
# plt.show()

g = sns.FacetGrid(data, row='Survived', col='Sex')
g.map(plt.hist, 'Age', alpha=0.7)
plt.show()

g = sns.FacetGrid(data, col='Survived')
g.map(plt.hist, 'Age', alpha=0.7)
plt.show()

g = sns.FacetGrid(data, col='Sex')
g.map(plt.hist, 'Age', alpha=0.7)
plt.show()
