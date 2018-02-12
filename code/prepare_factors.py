from pandas import read_csv, get_dummies

"""
Импорт данных
"""

titanic_df = read_csv("../data/tidy_data/train.csv")
test_df = read_csv("../data/tidy_data/test.csv")


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
titanic_df.drop(['Sex'], axis=1, inplace=True)
test_df.drop(['Sex'], axis=1, inplace=True)

# Создание фиктивных переменных
person_dummies_titanic = get_dummies(titanic_df['Person'])
person_dummies_test = get_dummies(test_df['Person'])
# добавление фиктивных переменных
titanic_df = titanic_df.join(person_dummies_titanic)
test_df = test_df.join(person_dummies_test)
# удаление ненужных категориальных столбцов
titanic_df.drop(['Person'], axis=1, inplace=True)
test_df.drop(['Person'], axis=1, inplace=True)

"""
Так же преобразуем категориальную переменную 'Pclass'. У нас будут три фиктивных столбца
"""
# создание фиктивных переменных
pclass_dummies_titanic = get_dummies(titanic_df['Pclass'])
pclass_dummies_test = get_dummies(test_df['Pclass'])
# задание читаемых названий столбцов
pclass_dummies_titanic.columns = ['Class_1', 'Class_2', 'Class_3']
pclass_dummies_test.columns = ['Class_1', 'Class_2', 'Class_3']
# добавление фиктивных переменных
titanic_df = titanic_df.join(pclass_dummies_titanic)
test_df = test_df.join(pclass_dummies_test)
# удаление ненужных категориальных столбцов
titanic_df.drop(['Pclass'], axis=1, inplace=True)
test_df.drop(['Pclass'], axis=1, inplace=True)

X_train = titanic_df.drop("Survived", axis=1)
Y_train = titanic_df["Survived"]
X_test = test_df.copy()

X_train.to_csv('../data/factors/x_train.csv', index=False)
Y_train.to_csv('../data/factors/y_train.csv', index=False)
X_test.to_csv('../data/factors/x_test.csv', index=False)
