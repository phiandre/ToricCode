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
from keras import backend as K
from keras import losses
import tensorflow as tf

# Class definition
class QNet:

	"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
	QNet class constructor.
		@param
			state_size: the size of the state part of the input.
			action_size: the size of the action part of the input.
	"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
	def __init__(self, state_size):

		K.set_image_dim_ordering('tf')
		# Save the state and action sizes as well as the overall input size.
		self.state_size = state_size
		# Define a Neural Network based on these parameters
		self.network = Sequential()
		self.network.add(Conv2D(512, (3,3), strides=(2,2), data_format = "channels_last" ,input_shape=[self.state_size, self.state_size, 1], activation='relu'))
		#self.network.add(Dense(10,input_shape=[self.state_size, self.state_size, 1], activation='relu'))
		self.network.add(Flatten())
		self.network.add(Dense(8, activation='relu'))
		self.network.add(Dense(8, activation='relu'))
		self.network.add(Dense(8, activation='relu'))

		self.network.add(Dense(4))
		self.network.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
		#print(self.network.summary())
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
	def improveQ(self, data, true_Q, individual = False):
		# Concatenate the state, action and true_Q data.
		# Improve the approximation of Q.
		if individual:
			BS = 1
		else:
			BS = data.shape[0]
		self.network.fit(data, true_Q, epochs=1, batch_size=BS, verbose=0)
	def gradCalc(self,state,Qtrue):
		outputTensor = self.network.output
		loss = losses.mean_squared_error(Qtrue,outputTensor)
		
		listOfVariableTensors = self.network.trainable_weights
		gradients = K.gradients(loss, listOfVariableTensors)
		sess = tf.InteractiveSession() #TF
		sess.run(tf.global_variables_initializer()) #TF
		evaluated_gradients = sess.run(gradients,feed_dict={self.network.input:state})
		sess.close()
		return evaluated_gradients