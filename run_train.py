import numpy as np
from RL import RLsys
from Env import Env
from GenerateToricData import Generate
from keras.models import load_model
import time
import os.path
import pickle


class MainClass:

	def __init__(self):
		
		self.alpha = -0.5 #epsilon decay
		
		self.loadNetwork = True #train an existing network
		self.networkName = 'Networks/BestNetwork47.h5' 
		
		self.saveRate = 99 #how often the network is saved

		# creates a new filename for numSteps each time we run the code
		self.getFilename()
		self.X=0
		self.n=0
		self.avgTol = 1000
		
		self.run()
		


	def getFilename(self):
		tmp = list('Steps/numSteps1.npy')
		self.filename = "Steps/"+'numSteps1.npy'
		self.static_element = 1

		while os.path.isfile("".join(tmp)):
			self.static_element += 1
			tmp[14] = str(self.static_element)
			self.filename = "".join(tmp)

	def rotateHumanRep(self,humanRep,j):
		tmp = np.concatenate([humanRep, humanRep[:,0:1]],axis=1)
		tmp1 = np.concatenate([tmp,tmp[0:1,:]])
		humanRep = np.rot90(tmp1,j)
		state = humanRep[0:(humanRep.shape[0]-1),0:(humanRep.shape[1]-1)]
		return state

	def run(self):
		actions = 4
		comRep = np.load('ToricCodeComputer.npy')
		humRep = np.load('ToricCodeHuman.npy')
		size = comRep.shape[0]
		
		rl = RLsys(actions, size)
		if self.loadNetwork:
			importNetwork = load_model(self.networkName)
			rl.qnet.network = importNetwork
		
		steps = np.zeros(comRep.shape[2]*4)
		averager = np.zeros(self.avgTol)
		
		
		trainingIteration = 0

		for i in range(comRep.shape[2]):
			for j in range(4):
				state = comRep[:,:,i]
				
				state = np.rot90(state,j)
				
				humanRep = humRep[:,:,i]
				humanRep = self.rotateHumanRep(humanRep,j)
				
				env = Env(state, humanRep, checkGroundState=True)
				numSteps = 0
				rl.epsilon = (1+trainingIteration)**(self.alpha)
				while len(env.getErrors()) > 0:
					numSteps = numSteps + 1
					observation = env.getObservation()
					observation2 = env.getObservation2()
					a, e = rl.choose_action(observation, observation2)
					r = env.moveError(a, e)
					new_observation = env.getObservation()
					new_observation2 = env.getObservation2()
					
					rl.learn(observation[:,:,e], observation2[:,:,e], a, r, new_observation, new_observation2)
				self.n += 1
				if (self.n) <= self.avgTol:
					if r == 5:
						averager[trainingIteration] = 1
					average = (np.sum(averager))/(self.n)
				else:
					if r == 5:
						avg1=averager[1:]
						avg2=np.ones((1))
						
						averager = np.concatenate((avg1, avg2))
					else:
						avg1 = averager[1:]
						avg2 = np.zeros((1))
						averager = np.concatenate((avg1, avg2))
					average = (np.sum(averager))/self.avgTol
				
				steps[trainingIteration] = numSteps
				
				
				
				
				print(' ')
				print('Episode ' + str(trainingIteration))
				print("Steps taken: "+ str(numSteps))
				if r ==5:
					print("Groundstate is RIGHT!")
				else:
					print("Groundstate is WRONG!")
				print("Average correct GS during last 500 episodes: " + str(100*average) + " %")

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


