"""""""""""""""""""""""""""""""""""""""""""""""""""

Policy for the approximation of the Q-function.

Utilizes a QNet object as the associated neural
network. Selects appropriate action based on an
epsilon-greedy policy.

"""""""""""""""""""""""""""""""""""""""""""""""""""

# Imports
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from QNet import QNet
import keras
import math
import numpy as np
import pandas as pd

# Class definition
class RLsys:

	"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
	RL class constructor.
		@param
			actions: the possible actions of the system.
			state_size: the size of the state matrix.
	"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
	def __init__(self, actions, state_size, reward_decay=0.9, e_greedy=0.9):
		# Save parameters for later use
		self.state_size = state_size
		self.actions = actions
		self.gamma = reward_decay
		self.epsilon = e_greedy
		# Produce neural network
		self.qnet = QNet(self.state_size)
		
	"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
	Method which returns the action based on specified state and error.
		@param
			observation: the current state of the system, centered
			around the errors. Dimensionality: NxNxE, where E is the
			amount of errors we wish to evaluate actions for.
		@return
			int: the given action based on the state.
			int: the associated error.
	"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
	def choose_action(self, observation):

		# ska returnera z-dimensionen
		numErrors = observation.shape[2]
		# de olika Q för alla errors
		predQ = self.predQ(observation)

		# Check the epsilon-greedy criterion
		if np.random.uniform() < self.epsilon:
			# Select the best action
			index = np.unravel_index(predQ.argmax(), predQ.shape)			
			# hämta det bästa action för ett visst error
			action = index[0]
			error = index[1]
		else:
			# Choose random action and error
			action = np.random.choice(self.actions)
			error = np.random.choice(range(numErrors))
			# slumpa error här
		
		# returnera action och error
		return action, error
		
	def predQ(self,observation):
		numErrors = observation.shape[2]
		# de olika Q för alla errors
		predQ = np.zeros([4, numErrors])
		# evaluera Q för de olika errors
		for x in range(numErrors):
			state = observation[:,:,x]
			predQ[:,x] = self.qnet.predictQ(state)
		return predQ
	"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
	Trains the neural network given the outcome of the action.
		@param
			state: the previous state of the system.
			action: the action taken.
			reward: the immediate reward received.
			observation_p: the resulting observation.
	"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
	def learn(self, state, action, reward):
		# Q is the more optimal Q
		Q = self.qnet.predictQ(state)[0,:]
		# Update the approximation of Q
		Q[action] = reward
		# Update the neural network
		self.qnet.improveQ(state, Q)

	"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
	Changes the epsilon in the epsilon-greedy policy.
		@param
			epsilon: the new epsilon.
	"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""       
	def changeEpsilon(self, epsilon):
		self.epsilon = epsilon


if __name__ == '__main__':

	rl = RLsys(4, 3)
	M = np.zeros([3, 3, 2])
	a, c = rl.choose_action(M)
	print(a)
	print(c)
