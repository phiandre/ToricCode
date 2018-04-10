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
		self.graphix = True
		self.saveData = True
		self.networkName = 'trainedNetwork13.h5'
		self.maxNumberOfIterations = 1000

		# creates a new filename each time we run the code
		if (self.saveData):
			tmp = list('/Users/nikfor/Desktop/Kandidat/Saves/numSteps1.npy')
			self.static_element = 1
			while os.path.isfile("".join(tmp)):
				self.static_element += 1
				tmp[45] = str(self.static_element)
			self.filename = "".join(tmp)


		self.run()
		
	def printState(self, env):
		print(env.state)
		print('')

	
        
	def run(self):

		importNetwork = load_model(self.networkName)

		rl = RLsys(4, importNetwork.input_shape[2])
		rl.qnet.network = importNetwork

		rl.changeEpsilon(1)
		humRep=np.load('ToricCodeHumanTest.npy')
		comRep=np.load('ToricCodeComputerTest.npy')
		print(comRep[:,:,3])

		iterations = np.zeros(comRep.shape[2])

		for i in range(min(comRep.shape[2],self.maxNumberOfIterations)):
			state=comRep[:,:,i]
			human=humRep[:,:,i]
			env = Env(state,human)
			numIter = 0
			while len(env.getErrors()) > 0:
				#print('Bana nummer ' + str(i))
				if self.graphix:
					self.printState(env)
				numIter = numIter + 1
				observation = env.getObservation()
				a, e = rl.choose_action(observation)
				r = env.moveError(a, e)
				new_observation = env.getObservation()

			print("Steps taken at iteration " +str(i) + ": ", numIter)
			iterations[i] = numIter
			




		print("Saving data in " + self.filename)
		np.save(self.filename,iterations)

        
       
       
"""""""""""""""""""""""""""""""""""""""""""""
Mainmetod, här körs själva simuleringen.
"""""""""""""""""""""""""""""""""""""""""""""
if __name__ == '__main__':
	MainClass()


