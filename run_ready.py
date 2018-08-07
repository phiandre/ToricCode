"""""""""""""""""""""
	Importeringar
"""""""""""""""""""""
import numpy as np
from RL import RLsys
from Env import Env
from BlossomEnv import Env as BEnv
from Blossom import Blossom
from GenerateToricData import Generate
from keras.models import load_model
import time
import os.path
import pickle

"""""""""""""""""""""""
	Klassdefinition
"""""""""""""""""""""""
class MainClass:

	def __init__(self):
		#TODO värden som skall sättas innan varje körning
		self.graphix = False
		self.saveData = False
		self.networkName = 'Networks/Network100kMEM.h5'
		self.maxNumberOfIterations = 10000
		
		self.X = 0
		self.n = 0
		# creates a new filename each time we run the code
		self.getFilename()
		self.run()

	def getFilename(self):
		if (self.saveData):
			tmp = list('/Users/nikfor/Desktop/Kandidat/Saves/numSteps1.npy')
			self.static_element = 1
			while os.path.isfile("".join(tmp)):
				self.static_element += 1
				tmp[45] = str(self.static_element)
			self.filename = "".join(tmp)
		else:
			tmp = list('numSteps1.npy')
			self.static_element = 1

			while os.path.isfile("".join(tmp)):
				self.static_element += 1
				tmp[8] = str(self.static_element)
				self.filename = "".join(tmp)



	def printState(self, env):
		print(env.state)
		print('')
		
	
	def printQ(self, observation, rl):
		predictedQ = rl.predQ(observation)
		for i in range(observation.shape[2]):
			print("state\n", observation[:,:,i])
			print("\n")
			print("upp: ", predictedQ[0,i])
			print("ner: ", predictedQ[1,i])
			print("vänster: ", predictedQ[2,i])
			print("höger: ", predictedQ[3,i])
			print("\n\n\n\n\n")
			

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
		size = 5
		importNetwork = load_model(self.networkName)

		rl = RLsys(4, importNetwork.input_shape[2])
		rl.qnet.network = importNetwork
		bCorr = 0
		largeNum = 0
		rl.changeEpsilon(0.2)
		humRep=np.load('ToricCodeHumanTest.npy')
		comRep=np.load('ToricCodeComputerTest.npy')
		print(comRep[:,:,3])
		#np.random.shuffle(comRep)
		iterations = np.zeros(comRep.shape[2])
		for i in range(min(comRep.shape[2],self.maxNumberOfIterations)):
			dist = 0
			state=comRep[:,:,i]
			human=humRep[:,:,i]
			"""
			if np.count_nonzero(state) > 0:
				state_ =np.copy(state)
				state_ = self.labelState(state_,size)
				BlossomObject = Blossom(state_)
				MWPM = BlossomObject.readResult()
				Benv = BEnv(state_, human, checkGroundState = True)
				for element in MWPM:
					error1 = element[0]+1
					error2 = element[1]+1
					blossomReward = Benv.blossomCancel(error1, error2)
			"""
			env = Env(state,human,checkGroundState=True)
			numIter = 0
			while len(env.getErrors()) > 0:
				#print('Bana nummer ' + str(i))
				if self.graphix:
					self.printState(env)
				numIter = numIter + 1
				observation = env.getObservation()
				#self.printQ(observation, rl)
				#print(env.state)
				print("ERRORS: ", len(env.getErrors()))
				a, e = rl.choose_action(observation)
				r = env.moveError(a, e)
				new_observation = env.getObservation()
				if numIter > 500:
					print(env.state)

			if r == env.correctGsR:
				self.X += 1

			if blossomReward == Benv.correctGsR:
				bCorr += 1


			self.n += 1
			print("CORRECT GROUNDSTATE:", self.X/self.n)
			print("Steps taken at iteration " +str(i) + ": ", numIter)

			print("MWPM correct ", bCorr / self.n)
			iterations[i] = numIter




		print("Saving data in " + self.filename)
		np.save(self.filename,iterations)

        
       
       
"""""""""""""""""""""""""""""""""""""""""""""
Mainmetod, här körs själva simuleringen.
"""""""""""""""""""""""""""""""""""""""""""""
if __name__ == '__main__':
	MainClass()


