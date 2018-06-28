import numpy as np
from RL import RLsys
from Env import Env
from GenerateToricData import Generate
from keras.models import load_model
import time
import os.path
import pickle
import math


class MainClass:
	
		


	def __init__(self):
		# Alla booleans
		self.loadNetwork = False #train an existing network
		self.gsRGrowth = True
		
		#Epsilon decay parameters
		
		self.alpha = -0.8 		# flyttar "änden" på epsilon-kurvan
		self.k = 20000			# flyttar "mitten" på epsilon-kurvan
		
		
		
		
		self.networkName = 'trainedNetwork42.h5' 
		
		self.saveRate = 100 #how often the network is saved

		# creates a new filename for numSteps each time we run the code
		self.getFilename()
		
		self.avgTol = 200 # Den mängd datapunkter som average tas över
		if self.gsRGrowth:
			self.fR = 5 # asymptotic reward
			self.w = math.pi/90000 #frequency (effects slope)
			self.b = -78000 # Phase shift (negative coordinate for center of slope)
		
		self.run()
		


		
		
		# Om man vill ha en tanh-kurveökning av Ground State Reward väljs parametrar här

		
		

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
		if self.loadNetwork:
			importNetwork = load_model(self.networkName)
			rl.qnet.network = importNetwork
		
		steps = np.zeros(comRep.shape[2]*4)
		
		averager = np.zeros(comRep.shape[2]*4) # Används till att räkna ut hur sannolikt algoritmen återvänder till rätt grundtillstånd
		
		n=0
		
		trainingIteration = 0
		if self.gsRGrowth:
			A = self.fR
			B = A
			w = self.w
			b = self.b
		for i in range(comRep.shape[2]):
			for j in range(4):
				state = comRep[:,:,i]
				
				state = np.rot90(state,j)
				
				humanRep = humRep[:,:,i]
				humanRep = self.rotateHumanRep(humanRep,j)
				
				env = Env(state, humanRep, checkGroundState=True)
				numSteps = 0
				rl.epsilon = ((self.k+trainingIteration)/self.k)**(self.alpha)
				if self.gsRGrowth:
					env.correctGsR = A*np.tanh(w*(trainingIteration+b)) + B
					
				while len(env.getErrors()) > 0:
					numSteps = numSteps + 1
					observation = env.getObservation()
					a, e = rl.choose_action(observation)
					r = env.moveError(a, e)
					new_observation = env.getObservation()
					rl.learn(observation[:,:,e], a, r, new_observation)
				
				
				if r == env.correctGsR:
					averager[n] = 1
				n += 1
				
				if n < self.avgTol:
					average = np.sum(averager)/n
				else:
					average = np.sum(averager[(n-self.avgTol):n])/self.avgTol
				print("Steps taken at iteration " +str(trainingIteration) + ": ", numSteps)
				if n<self.avgTol:
					print("Probability of correct GS last " + str(n) + ": " + str(average*100) + " %")
				else:
					print("Probability of correct GS last " + str(self.avgTol) + ": " + str(average*100) + " %")
				steps[trainingIteration] = numSteps
				
				

				if((trainingIteration+1) % self.saveRate == 0):
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


