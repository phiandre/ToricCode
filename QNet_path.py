"""""""""""""""""""""""""""""""""""""""""""""""""""

Neural network for approximation of Q-function.

Will take in concatenated state- and action data
and ouput the associated approximation Q(s,a).

"""""""""""""""""""""""""""""""""""""""""""""""""""

# Imports
from keras.models import Sequential, Model
from keras.layers import Dense, Input, concatenate
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
		#sizePick = np.zeros(2)
		
		#BRANCH 1: STATE -->{LAYERS}\
		#			MERGED			 = > {More layers and output}
		#BRANCH 2: MEMORY-->{LAYERS}/
		
		#Branch 1
		state_input = Input(shape=(self.state_size,self.state_size,1),name = 'state_input')
		branch1 = Conv2D(512, (3,3), strides = (2,2), data_format = 'channels_last')(state_input)
		branch1 = Flatten()(branch1)
		
		#Branch 2
		path_hor_input = Input(shape=(self.state_size,self.state_size,1),name = 'path_hor_input')
		branch2 = Conv2D(40, (state_size,1), strides = (1,1), data_format = 'channels_last')(path_hor_input)
		branch2 = Flatten()(branch2)

		#Branch 1
		path_vert_input=Input(shape=(self.state_size,self.state_size,1),name = 'path_vert_input')
		branch3 = Conv2D(40, (1,state_size), strides = (1,1), data_format = 'channels_last')(path_vert_input)
		branch3 = Flatten()(branch3)
		
		#Merged branch
		mergeBranch = concatenate([branch1,branch2,branch3])
		mergeBranch = Dense(50, activation = 'relu')(mergeBranch)
		mergeBranch = Dense(40, activation = 'relu')(mergeBranch)
		mergeBranch = Dense(30, activation = 'relu')(mergeBranch)
		
		"""
		#mergeBranch = Dropout(0.2)(mergeBranch)
		mergeBranch = Dense(state_size**2, activation = 'relu')(mergeBranch)
		#mergeBranch = Dropout(0.2)(mergeBranch)
		mergeBranch = Dense(int(max(np.ceil(0.8*state_size**2),6)), activation = 'relu')(mergeBranch)
		mergeBranch = Dense(int(max(np.ceil(0.6*state_size**2),6)), activation = 'relu')(mergeBranch)
		mergeBranch = Dense(int(max(np.ceil(0.4*state_size**2),6)), activation = 'relu')(mergeBranch)
		"""
		Q_output = Dense(4, name = 'Q_output')(mergeBranch)
		
		self.network = Model(inputs = [state_input, path_hor_input, path_vert_input], outputs=Q_output)
		self.network.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
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
		state_1 = state[:,:,0]
		data1 = state_1[np.newaxis,:,:,np.newaxis]
		state_2 = state[:,:,1]
		data2 = state_2[np.newaxis,:,:,np.newaxis]
		state_3 = state[:,:,2]
		data3 = state_3[np.newaxis,:,:,np.newaxis]
		# Predict the value of Q
		return self.network.predict([data1,data2,data3])

	"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
	Method which improves the value-approximation of Q.
		@param
			state: the current state of the system.
			action: the action to be explored.
			true_Q: the better approximative value of Q.
	"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
	def improveQ(self, state, true_Q):
		# Concatenate the state, action and true_Q data.
		data1 = state[:,:,:,0]
		data1 = data1[:,:,:,np.newaxis]
		data2 = state[:,:,:,1]
		data2 = data2[:,:,:,np.newaxis]
		data3 = state[:,:,:,2]
		data3 = data3[:,:,:,np.newaxis]
		true_Q = true_Q
		# Improve the approximation of Q.
		self.network.fit([data1, data2, data3], true_Q, epochs=1, batch_size=1, verbose=0)