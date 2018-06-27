"""""""""""""""""""""
	Importeringar
"""""""""""""""""""""
import numpy as np
from RL import RLsys
from Env import Env
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
		self.networkName = 'Networks/trainedNetwork16.h5'
		self.maxNumberOfIterations = 10000

		# creates a new filename each time we run the code
		self.getFilename()
		self.n = 0
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
			
		

	
        
	def run(self):

		importNetwork = load_model(self.networkName)
		for layer in importNetwork.layers:
			print(layer.get_weights())
		
		quit()
		
		rl = RLsys(4, importNetwork.input_shape[2])
		rl.qnet.network = importNetwork
		
		largeNum = 0
		rl.changeEpsilon(0)
		humRep=np.load('ToricCodeHumanTest.npy')
		comRep=np.load('ToricCodeComputerTest.npy')
		averager = np.zeros(humRep.shape[2])
		
		print(comRep[:,:,3])
		#np.random.shuffle(comRep)
		iterations = np.zeros(comRep.shape[2])
		for i in range(min(comRep.shape[2],self.maxNumberOfIterations)):
			state=comRep[:,:,i]
			human=humRep[:,:,i]
			env = Env(state,human, checkGroundState = True)
			numIter = 0
			while len(env.getErrors()) > 0:
				#print('Bana nummer ' + str(i))
				if self.graphix:
					self.printState(env)
				numIter = numIter + 1
				observation = env.getObservation()
				self.printQ(observation, rl)
				
				a, e = rl.choose_action(observation)
				r = env.moveError(a, e)
				new_observation = env.getObservation()
			if r == env.cGS:
				averager[self.n] = 1
			self.n += 1
			average = np.sum(averager)/self.n
			
			print("Steps taken at iteration " +str(i+1) + ": ", numIter)
			iterations[i] = numIter
			print("Probability of correct groundstate: " + str(average*100) + " %")
			print(' ')




		print("Saving data in " + self.filename)
		np.save(self.filename,iterations)

        
       
       
"""""""""""""""""""""""""""""""""""""""""""""
Mainmetod, här körs själva simuleringen.
"""""""""""""""""""""""""""""""""""""""""""""
if __name__ == '__main__':
	MainClass()


