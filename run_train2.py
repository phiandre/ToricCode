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
		
		self.alpha = 0 # epsilon decay
		#train an existing network
		self.loadNetwork = False
		self.networkName = 'Networks/trainedNetwork15.h5'
		
		#The acting network
		self.Rl1networkName = 'Networks/trainedNetwork15.h5'
		
		self.saveRate = 99 #how often the network is saved

		# creates a new filename for numSteps each time we run the code
		self.getFilename()
		
		self.avgTol = 100000 # Den mängd datapunkter som average tas över
		
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
		humRep=np.load('ToricCodeHuman.npy')
		size = comRep.shape[0]
		
		rl = RLsys(actions, size)
		rl2 = RLsys(actions, size)
		if self.loadNetwork:
			importNetwork = load_model(self.networkName)
			rl2.qnet.network = importNetwork
		RL1importNetwork = load_model(self.Rl1networkName)
		rl.qnet.network = RL1importNetwork
		
		steps = np.zeros(comRep.shape[2]*4)
		
		averager = np.zeros(comRep.shape[2]*4) # Används till att räkna ut hur sannolikt algoritmen återvänder till rätt grundtillstånd
		
		n=0
		
		trainingIteration = 0

		for i in range(comRep.shape[2]):
			for j in range(4):
				state = comRep[:,:,i]
				
				state = np.rot90(state,j)
				
				humanRep = humRep[:,:,i]
				humanRep = self.rotateHumanRep(humanRep,j)
				
				env = Env(state, humanRep, checkGroundState=True)
				numSteps = 0
				rl.epsilon = 0.16 #(1+trainingIteration)**(self.alpha)
				
				while len(env.getErrors()) > 0:
					numSteps = numSteps + 1
					observation = env.getObservation()
					a, e = rl.choose_action(observation)
					r = env.moveError(a, e)
					new_observation = env.getObservation()
					rl2.learn(observation[:,:,e], a, r, new_observation)
				
				if r == env.cGS:
					averager[n] = 1
				n += 1
				
				if n < self.avgTol:
					average = np.sum(averager)/n
				else:
					average = np.sum(averager[(n-self.avgTol):n])/self.avgTol
				print("Steps taken at iteration " +str(trainingIteration) + ": ", numSteps)
				if n < self.avgTol:
					print("Probability of correct GS last " + str(n) + ": " + str(average*100) + " %")
				else:
					print("Probability of correct GS last " + str(self.avgTol) + ": " + str(average*100) + " %")
				steps[trainingIteration] = numSteps
				
				

				if(trainingIteration % self.saveRate == 0):
					tmp = list('Networks/trainedNetwork1.h5')
					tmp[23] = str(self.static_element)
					filename = "".join(tmp)

					np.save(self.filename,steps[0:(trainingIteration+1)])

					rl2.qnet.network.save(filename)
					
				trainingIteration = trainingIteration + 1


		

"""""""""""""""""""""""""""""""""""""""""""""
Mainmetod, här körs själva simuleringen.
"""""""""""""""""""""""""""""""""""""""""""""
if __name__ == '__main__':
	MainClass()

