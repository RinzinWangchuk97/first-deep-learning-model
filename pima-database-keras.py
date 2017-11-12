# -*- coding: utf-8 -*-
"""
Created on Thu Nov  9 08:41:02 2017

@author: rinziii
"""

# set the working directory
from os import chdir
chdir('C:\\Users\\rinziii\\Documents\\codes_that_works')

import numpy as np
import pandas as pd


from keras.models import Sequential
from keras.layers import Dense

from sklearn.model_selection import train_test_split
# load dataset
dataset = np.array(pd.read_csv('pima\\pima-indians-diabetes.csv'))
X = dataset[:, 0:8]
Y = dataset[:, 8]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33, random_state=42)


# defining the model. Dense model means fully connected layer
model = Sequential()
model.add(Dense(12, input_dim=8, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# complining the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# fitting and training the model on my data
model.fit(X_train, Y_train, epochs=300, batch_size=10)
# evaluate the model
train_scores = model.evaluate(X_train, Y_train)
test_scores = model.evaluate(X_test, Y_test)

print('train_accuracy', train_scores[1]*100)
print('test_accuracy', test_scores[1] * 100)