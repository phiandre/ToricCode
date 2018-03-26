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
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    def __init__(self, actions, reward_decay=0.9, e_greedy=0.9, state_size=4):
        # Save parameters for later use
        self.state_size = state_size
        self.actions = actions
        self.gamma = reward_decay
        self.epsilon = e_greedy
        # Produce neural network
        self.qnet = QNet(self.state_size, 1)

    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    Method which returns the action based on specified state.
        @param
            observation: the current state of the system.
        @return
            int: the given action based on the state.
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    def choose_action(self, observation):
        # Check the epsilon-greedy criterion
        if np.random.uniform() < self.epsilon:
            # Produce Q for all possible actions
            predQ = np.zeros(len(self.actions))
            # Do predictions for each action
            for a in range(0, len(self.actions)):
                # Predict action with Neural Network
                predQ[a] = self.qnet.predictQ(observation, a)
            # Select the best action
            action = predQ.argmax()
        else:
            # Choose random action
            action = np.random.choice(self.actions)
        return action

    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    Trains the neural network given the outcome of the action.
        @param
            state: the previous state of the system.
            action: the action taken.
            reward: the reward received.
            state_: the resulting state.
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    def learn(self, state, action, reward, state_):
        # Check if we are at terminal state
        if state_ != 'terminal':
            # Produce Q based on possible actions
            predQ = np.zeros(len(self.actions))
            # Do predictions for each action
            for a in range(0, len(self.actions)):
                # Predict action with Neural Network
                predQ[a] = self.qnet.predictQ(state_, a)
            # Update the approximation of Q
            q_target = reward + self.gamma * predQ.max()
        else:
            # Update the approximation of Q
            q_target = reward
        # Update the neural network
        self.qnet.improveQ(state, action, q_target)

    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    Changes the epsilon in the epsilon-greedy policy.
        @param
            epsilon: the new epsilon.
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""       
    def changeEpsilon(self, epsilon):
        self.epsilon = epsilon

