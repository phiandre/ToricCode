"""""""""""""""""""""""""""""""""""""""""""""""""""

Neural network for approximation of Q-function.

Will take in concatenated state- and action data
and ouput the associated approximation Q(s,a).

"""""""""""""""""""""""""""""""""""""""""""""""""""

# Imports
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import Conv2D
from keras import optimizers
import numpy as np

# Class definition
class QNet:

	"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
	QNet class constructor.
		@param
			state_size: the size of the state part of the input.
			action_size: the size of the action part of the input.
	"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
	def __init__(self, state_size):
		# Save the state and action sizes as well as the overall input size.
		self.state_size = state_size
		# Define a Neural Network based on these parameters
		self.network = Sequential()
		self.network.add(Conv2D(50, 3, input_shape=[self.state_size, self.state_size,1], activation='relu'))
		self.network.add(Flatten())
		self.network.add(Dropout(0.2))
		self.network.add(Dense(100, activation='relu'))
		self.network.add(Dropout(0.2))
		self.network.add(Dense(100, activation='relu'))
		self.network.add(Dropout(0.2))
		self.network.add(Dense(100, activation='relu'))
		self.network.add(Dropout(0.2))
		self.network.add(Dense(100, activation='relu'))
		self.network.add(Dropout(0.2))
		self.network.add(Dense(100, activation='relu'))
		self.network.add(Dropout(0.2))
		self.network.add(Dense(100, activation='relu'))
		self.network.add(Dropout(0.2))
		self.network.add(Dense(100, activation='relu'))
		self.network.add(Dense(4, activation='sigmoid'))
		self.network.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
		
		trainingData = np.load('ToricCodeComputer200k10p.npy')
		trainingLabel = np.load('Label_200k10p.npy')
		print(trainingLabel[0:10,:])
		trainingData = trainingData[:,:,:,np.newaxis]

		testData = np.load('ToricCodeComputerTest200k10p.npy')
		testLabel = np.load('LabelTest200k10p.npy')
		
		self.network.fit(trainingData, trainingLabel, epochs = 50, batch_size = 10000)

		counter = np.zeros(4)
		
		correctGS = 0
		for i in range(testData.shape[0]):
			data = np.zeros((1,5,5,1))
			data[0,:,:,0] = testData[i,:,:]
			predict = self.network.predict(data)
			print(predict)
			groundState = np.argmax(predict)

			counter[groundState] += 1

			correct = np.argmax(testLabel[i,:])
			
			if groundState == correct:
				correctGS += 1
			
			print("Correct GS: ", correctGS/(i+1))
			for j in range(4):
				print("Guessed groundstate " + str(j) + ": " + str(counter[j]) + " times")

			
		#score = self.network.evaluate(testData, testLabel, batch_size = 10)
		
		#print("score: ", score)
		

	"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
	Method which predicts the Q associated which state and action.
		@param
			state: the current state of the system.
			action: the action to be explored.
		@return
			float: the value of the predicted Q.
	"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
	def predictQ(self, state):
		# Concatenate the state and action data.
		data = np.expand_dims(state, axis=0)
		# Predict the value of Q
		return self.network.predict(data)

	"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
	Method which improves the value-approximation of Q.
		@param
			state: the current state of the system.
			action: the action to be explored.
			true_Q: the better approximative value of Q.
	"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
	def improveQ(self, state, true_Q):
		# Concatenate the state, action and true_Q data.
		data = np.expand_dims(state, axis=0)
		true_Q = np.expand_dims(true_Q, axis=0)
		# Improve the approximation of Q.
		self.network.fit(data, true_Q, epochs=1, batch_size=1, verbose=0)

if __name__ == '__main__':
	
	QNet(5)
