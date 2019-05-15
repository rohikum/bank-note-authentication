import numpy as np
import pandas as pd

from keras.models import Sequential 
from keras.layers import Dense
from keras.utils import to_categorical

from sklearn.preprocessing import StandardScaler 
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score

dataframe = pd.read_csv("/home/rohikum/Desktop/data.txt", header=None)

#store the numerical content from this dataframe
dataset = dataframe.values 

#split this dataframe into input (X) and output values (Y)
X = dataset[:,0:4].astype(float)
Y = dataset[:,4]

#train-test split
seed = 0
[X_train, X_test, Y_train, Y_test] = train_test_split(X, Y, test_size = 0.30, random_state = seed)

scaler = StandardScaler().fit(X_train)

X_train = scaler.transform(X_train)

X_test = scaler.transform(X_test)

model = Sequential()

#First hidden layer which accepts input from the input layer
model.add(Dense(60, input_dim = 4, activation = 'sigmoid'))
#Second hidden layer
model.add(Dense(20, activation = 'sigmoid'))
#output layer
model.add(Dense(1, activation = 'sigmoid'))

#compile the neural network structure
model.compile(loss='binary_crossentropy', optimizer = 'adam', metrics=['accuracy'])

#fitting the model using the train data with a validation set of 33%
model.fit(X_train, Y_train, validation_split = 0.33, epochs = 100)

#making ouptput predictions based on the test set
y_pred = model.predict(X_test)

#evaluate against the test data_set
score = model.evaluate(X_test, Y_test, verbose = 1)

#print the accuracy
print("Accuracy of Model:", round(score[1]*100, 2), "%")
