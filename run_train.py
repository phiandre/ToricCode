import numpy as np
from RL import RLsys
from Env import Env
from GenerateToricData import Generate
from keras.models import load_model
import time
import os.path
import pickle
from collections import deque


class MainClass:

	def __init__(self):
		
		self.alpha = -0.5 #epsilon decay
		
		self.loadNetwork = False #train an existing network
		self.networkName = 'trainedNetwork42.h5' 
		
		self.saveRate = 99 #how often the network is saved

		# creates a new filename for numSteps each time we run the code
		self.getFilename()
		
		self.nsarsa()


	def getFilename(self):
		tmp = list('Steps/numSteps1.npy')
		self.filename = "Steps/"+'numSteps1.npy'
		self.static_element = 1

		while os.path.isfile("".join(tmp)):
			self.static_element += 1
			tmp[14] = str(self.static_element)
			self.filename = "".join(tmp)
			
	
	def nsarsa(self):
		actions = 4
		comRep = np.load('ToricCodeComputer.npy')
		size = comRep.shape[0]
		
		
		rl = RLsys(actions, size)
		
		steps = np.zeros(comRep.shape[2]*4)
		
		trainingIteration = 0
		
		
		for i in range(comRep.shape[2]):
			state = comRep[:,:,i]
			env = Env(state)
			rewardList = deque()
			actionList = deque()
			stateList = deque()
			T = 1000000
			n = 1
			tau = 0
			t = -1
			numSteps = 0
			
			observation = env.getObservation()
			a, e = rl.choose_action(observation)
			
			stateList.append(observation[:,:,e])
			actionList.append(a)
			
			while tau < (T-1):
				
				if t<T:
					t = t+1
					r = env.moveError(a, e)
					new_observation = env.getObservation()
					rewardList.append(r)
					if new_observation == 'terminal':
						T = t+1
						stateList.append(new_observation)
					else:
						a, e = rl.choose_action(new_observation)
						stateList.append(new_observation[:,:,e])
				tau = t-n+1
				stateArray = np.asarray(stateList)
				actionArray = np.asarray(actionList)
				rewardArray = np.asarray(rewardList)
				if tau >=0:
					rl.learn(stateArray, actionArray, rewardArray, tau, n, T, new_observation)
				numSteps = numSteps + 1	
				
			trainingIteration = trainingIteration + 1
					

			print("Steps taken at iteration " +str(trainingIteration) + ": ", numSteps)
					
			
			



	def run(self):
		actions = 4
		comRep = np.load('ToricCodeComputer.npy')
		size = comRep.shape[0]
		
		rl = RLsys(actions, size)
		if self.loadNetwork:
			importNetwork = load_model(self.networkName)
			rl.qnet.network = importNetwork
		
		steps = np.zeros(comRep.shape[2]*4)
		
		trainingIteration = 0

		for i in range(comRep.shape[2]):
			for j in range(4):
				state = comRep[:,:,i]
				
				state = np.rot90(state,j)
				
				env = Env(state)
				numSteps = 0
				rl.epsilon = (1+trainingIteration)**(self.alpha)
				
				T = 1000000
				n = 1
				tau = 0
				
				rewardList = deque()
				actionList = deque()
				stateList = deque()
								
				while tau < (T-1):
					if numSteps < T:
						observation = env.getObservation()
						a, e = rl.choose_action(observation)
						r = env.moveError(a, e)
						rewardList.append(r)
						actionList.append(a)
						stateList.append(observation[:,:,e])
						new_observation = env.getObservation()
						
						if new_observation == 'terminal':
							T = numSteps + 1
					tau = numSteps - n + 1
					
					stateArray = np.asarray(stateList)
					actionArray = np.asarray(actionList)
					rewardArray = np.asarray(rewardList)
					if tau >= 0:
						rl.learn(stateArray, actionArray, rewardArray, tau, n, T, new_observation)
							
						
						
					numSteps = numSteps + 1

				print("Steps taken at iteration " +str(trainingIteration) + ": ", numSteps)
				steps[trainingIteration] = numSteps

				if(trainingIteration % self.saveRate == 0):
					tmp = list('Networks/trainedNetwork1.h5')
					tmp[23] = str(self.static_element)
					filename = "".join(tmp)

					np.save(self.filename,steps[0:(trainingIteration+1)])

					rl.qnet.network.save(filename)
					
				trainingIteration = trainingIteration + 1


		

"""""""""""""""""""""""""""""""""""""""""""""
Mainmetod, här körs själva simuleringen.
"""""""""""""""""""""""""""""""""""""""""""""
if __name__ == '__main__':
	MainClass()


