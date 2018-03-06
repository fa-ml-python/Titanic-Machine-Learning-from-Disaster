from pandas import read_csv, concat
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

X = read_csv("data/factors/x_train.csv")
Y = read_csv("data/factors/y_train.csv")
data = concat([Y, X], axis=1)

train, test = train_test_split(data, test_size=0.3, random_state=0, stratify=data['Survived'])
X_train = train[train.columns[1:]]
Y_train = train[train.columns[:1]]
X_test = test[test.columns[1:]]
Y_test = test[test.columns[:1]]

print(X_train.shape)

model = LogisticRegression()
model.fit(X_train, Y_train)
print(model.score(X_test, Y_test))


from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.models import Sequential

# create first model
model = Sequential()
model.add(Dense(18, input_dim=18, activation='relu'))
model.add(Dense(18, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
# Compile model
model.compile(loss='binary_crossentropy', optimizer='Nadam', metrics=['accuracy'])
# Fit the model
model.fit(X_train, Y_train, epochs=1000, batch_size=32)
# evaluate the model
scores = model.evaluate(X, Y)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
