import pandas as pd
import numpy as np
import sklearn
# import matpl
import pickle
from sklearn import linear_model

# read in files
data = pd.read_csv("student-mat.csv", sep=";")

data = data[['G1', 'G2', 'G3', 'studytime', 'failures', 'absences']]

predict = 'G3'

# features
X = np.array(data.drop([predict], 1))

# all labels
y = np.array(data[predict])

# splitting into test train
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.1)

# create model
model = linear_model.LinearRegression()

# fit data into model
model.fit(x_train, y_train)

acc = model.score(x_test, y_test)
print(acc)

# save model
with open('samplemodel.pickle', 'wb') as f:
    pickle.dump(model, f)

# printing coefficient and intercept
print('Intercept: ', + model.intercept_)
print('Coefficient: ', + model.coef_)

# making predictions
predictions = model.predict(x_test)

# print out values
for x in range(len(predictions)):
    print(predictions[x], x_test[x], y_test[x])