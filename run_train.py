import numpy as np
from RL import RLsys
from Env import Env
from GenerateToricData import Generate
from keras.models import load_model
import time
import os.path
import pickle
import math
from Blossom import Blossom 
import time

class MainClass:
	
		


	def __init__(self):
		# Alla booleans
		self.loadNetwork = False #train an existing network
		self.gsRGrowth = np.load("Tweaks/GSgrowth.npy")
		self.windowsize = 7
		
		self.checkGS = np.load("Tweaks/checkGS.npy")
		
		#Epsilon decay parameters
		
		self.epsilonDecay = np.load("Tweaks/epsilonDecay.npy")
		if self.epsilonDecay:
			self.alpha = np.load("Tweaks/alpha.npy") 		# flyttar "änden" på epsilon-kurvan
			self.k = np.load("Tweaks/k.npy")			# flyttar "mitten" på epsilon-kurvan
		
		
		
		self.networkName = 'trainedNetwork42.h5' 
		
		self.saveRate = 2 #how often the network is saved

		# creates a new filename for numSteps each time we run the code
		self.getFilename()
		
		self.avgTol = 1000 # Den mängd datapunkter som average tas över
		self.fR = np.load("Tweaks/correctGsR.npy") # asymptotic Ground State reward
		
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
		
	
	def labelState(self, s, size):
		state = s
		label = 1
		for j in range(size):
			for k in range(size):
				if state[j,k] == 1:
					state[j,k] = label
					label +=1
		return state
		
	def run(self):
		actions = 4
		comRep = np.load('ToricCodeComputer.npy')
		humRep=np.load('ToricCodeHuman.npy')
		size = comRep.shape[0]
		

		rl = RLsys(actions, size, windowSize = self.windowsize)
		if self.loadNetwork:
			importNetwork = load_model(self.networkName)
			rl.qnet.network = importNetwork

		steps = np.zeros(comRep.shape[2]*4)

		averager = np.zeros(comRep.shape[2]*4) # Används till att räkna ut hur sannolikt algoritmen återvänder till rätt grundtillstånd

		n=0

		rl.epsilon = np.load("Tweaks/epsilon.npy")
		rl.gamma = 0.9

		trainingIteration = 0
		if self.gsRGrowth:
			A = np.load("Tweaks/AGS.npy")
			B = np.load("Tweaks/BGS.npy")
			w = np.load("Tweaks/wGS.npy")
			b = np.load("Tweaks/bGS.npy")

		incorrectGsR = np.load("Tweaks/incorrectGsR.npy")
		stepR = np.load("Tweaks/stepR.npy")

		segmentSize = 3

		rl2 = RLsys(actions, int(size/segmentSize), windowSize=self.windowsize)






		for i in range(comRep.shape[2]):
			for j in range(4):
				state = np.copy(comRep[:,:,i])
				state = np.rot90(state,j)

				humanRep = humRep[:,:,i]
				humanRep = self.rotateHumanRep(humanRep,j)

				env = Env(state, humanRep, checkGroundState=self.checkGS, segmentSize=segmentSize, windowSize= self.windowsize)
				env.incorrectGsR = incorrectGsR
				env.stepR = stepR
				numSteps = 0

				if self.epsilonDecay:
					rl.epsilon = ((self.k+trainingIteration+12000)/self.k)**(self.alpha)
				if self.gsRGrowth:
					env.correctGsR = A*np.tanh(w*(trainingIteration+b)) + B
				else:
					env.correctGsR = self.fR
				r = 0
				alone = False
				env.elimminationR = 1

				while (not alone) and (len(env.getErrors()) > 0):
					numSteps = numSteps + 1
					observation, x, indexVector = env.getObservation()
					a, e = rl.choose_action(observation, indexVector)
					#print("error ", e)
					r = env.moveError(a, e)
					new_observation, alone, newIndexVector = env.getObservation()
					#print('State: \n', env.state)
					"""
					if alone:
						r = len(env.getErrors()) * -2.5
					"""
					#rl.storeTransition(observation[:,:,list(indexVector).index(e)], a, r, new_observation)
					rl.storeTransition(observation[list(indexVector).index(e),:,:], a, r, new_observation)
					rl.learn(alone)

					if(numSteps % 20 == 0):
						print("Errors remaining: ", len(env.getErrors()))

				print("Steps taken at iteration " + str(trainingIteration) + ": ", numSteps)
				print("Errors remaining: ", len(env.getErrors()))
				"""
				print("I am zooming out...")
				zoomedOutState = env.zoomOut()
				zoomedOutEnv = Env(zoomedOutState, windowSize=7)
				zoomedOutEnv.elimminationR = -1
				#print("Zoomed out state \n", zoomedOutEnv.state)

				while len(env.getErrors()) > 0:

					#print("zoomedOutEnv\n", zoomedOutEnv.state)
					#print("env\n", env.state)
					#print("len", len(env.getErrors()))
					observation, x, indexVector = zoomedOutEnv.getObservation()

					a, e = rl2.choose_action(observation, indexVector)
					index = zoomedOutEnv.getErrors()[e,:]


					nextPos = zoomedOutEnv.getPos(a, index)

					annihilation = (zoomedOutEnv.state[nextPos[0],nextPos[1]]==1)
					r2 = zoomedOutEnv.moveError(a, e)
					longStep = env.longMove(a,index)
					numSteps += longStep
					if annihilation:
						pairSteps, reward = env.pairErrors(nextPos)
						numSteps += pairSteps



					new_observation, alone, newindexVector = zoomedOutEnv.getObservation()

					rl2.storeTransition(observation[list(indexVector).index(e),:, :], a, r2, new_observation)
					rl2.learn(alone=False)



					if (numSteps % 20 == 0):
						print("Errors remaining: ", len(env.getErrors()))

				



				
				if self.checkGS:
					if r != 0:
						if r == env.correctGsR:
							averager[n] = 1
						n += 1
					
					if n < self.avgTol:
						average = np.sum(averager)/n
					else:
						average = np.sum(averager[(n-self.avgTol):n])/self.avgTol
				
				
				
				if self.checkGS:
					if n<self.avgTol:
						print("Probability of correct GS last " + str(n) + ": " + str(average*100) + " %")
					else:
						print("Probability of correct GS last " + str(self.avgTol) + ": " + str(average*100) + " %")
					steps[trainingIteration] = numSteps
				"""

				if((trainingIteration+1) % self.saveRate == 0):

					tmp = list('Networks/trainedNetwork1.h5')
					tmp[23] = str(self.static_element)
					filename = "".join(tmp)

					#np.save(self.filename,steps[0:(trainingIteration+1)])

					rl.qnet.network.save(filename)





				trainingIteration = trainingIteration + 1


		

"""""""""""""""""""""""""""""""""""""""""""""
Mainmetod, här körs själva simuleringen.
"""""""""""""""""""""""""""""""""""""""""""""
if __name__ == '__main__':
	MainClass()


