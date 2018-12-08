import numpy as np
import csv
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras.datasets import mnist
from keras import backend as K
import matplotlib  
from keras.utils import plot_model
matplotlib.use('TkAgg') 
#from matplotlib import pyplot as plt
np.random.seed(123)

acc = []
for num in ['4']:
	X_train = []
	Y_train = []

	X_test = []
	Y_test = []

	with open('data/training_data' + num + '.csv') as csv_file:
		csv_reader = csv.reader(csv_file, delimiter=',')
		for row in csv_reader:
			Y_train.append(row[-1])
			X_train.append(row[:-1])

	with open('data/testing_data' + num + '.csv') as csv_file:
		csv_reader = csv.reader(csv_file, delimiter=',')
		for row in csv_reader:
			Y_test.append(row[-1])
			X_test.append(row[:-1])


	num_attr = len(X_train[0])

	X_train = np.asarray(X_train, dtype=np.float32)
	X_test = np.asarray(X_test, dtype=np.float32)

	Y_train = np.asarray(Y_train, dtype=np.float32)
	Y_test = np.asarray(Y_test, dtype=np.float32)
	Y_train = np_utils.to_categorical(Y_train, 2) 
	Y_test = np_utils.to_categorical(Y_test, 2)



	# create model
	model = Sequential()
	model.add(Dense(20, input_dim=num_attr, activation='relu'))
	model.add(Dense(10, activation='relu'))
	model.add(Dense(2, activation='softmax'))   #need change


	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

	# Fit the model
	model.fit(X_train, Y_train, epochs=50, batch_size=50)

	scores = model.evaluate(X_test, Y_test)
	print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
	acc.append(scores[1]*100)
print(acc)
print(model.get_weights())
