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
		self.networkName = 'Networks/trainedNetwork55.h5'
		self.maxNumberOfIterations = 10000

		# creates a new filename each time we run the code
		self.getFilename()
		self.run()

	def getFilename(self):

		tmp = list('numSteps1.npy')
		self.static_element = 1

		while os.path.isfile("".join(tmp)):
			self.static_element += 1
			tmp[8] = str(self.static_element)
			self.filename = "".join(tmp)

	def printQ(self, observation, observation2, rl):
		predictedQ = rl.predQ(observation, observation2)
		for i in range(observation.shape[2]):
			print("state\n", observation[:,:,i])
			print("\n")
			print("upp: ", predictedQ[0,i])
			print("ner: ", predictedQ[1,i])
			print("vänster: ", predictedQ[2,i])
			print("höger: ", predictedQ[3,i])
			print("\n\n\n\n\n")

	def printState(self, env):
		print(env.state)
		print('')

	
        
	def run(self):
		humRep=np.load('ToricCodeHumanTest.npy')
		comRep=np.load('ToricCodeComputerTest.npy')
		
		size = comRep.shape[0]
		
		importNetwork = load_model(self.networkName)

		rl = RLsys(4, size)
		rl.qnet.network = importNetwork
		
		largeNum = 0
		rl.changeEpsilon(0)
		humRep=np.load('ToricCodeHumanTest.npy')
		comRep=np.load('ToricCodeComputerTest.npy')
		
		n=0
		
		averager = np.zeros(comRep.shape[2])
		
		iterations = np.zeros(comRep.shape[2])
		for i in range(min(comRep.shape[2],self.maxNumberOfIterations)):
			state=comRep[:,:,i]
			human=humRep[:,:,i]
			env = Env(state,human, checkGroundState=True)
			numIter = 0
			while len(env.getErrors()) > 0:
				#print('Bana nummer ' + str(i))
				if self.graphix:
					self.printState(env)
				numIter = numIter + 1
				observation = env.getObservation()
				observation2 = env.getObservation2()
				#self.printQ(observation, observation2, rl)
				
				a, e = rl.choose_action(observation, observation2)
				r = env.moveError(a, e)

			if numIter > 50:
				largeNum = largeNum + 1
			if r == 5:
				averager[n] = 1
			n += 1
			average=np.sum(averager)/n
			
			print("Steps taken at iteration " +str(i) + ": ", numIter)
			print("Average Groundstate: " + str(np.sum(average)))
			
			




		print("Saving data in " + self.filename)
		np.save(self.filename,iterations)

        
       
       
"""""""""""""""""""""""""""""""""""""""""""""
Mainmetod, här körs själva simuleringen.
"""""""""""""""""""""""""""""""""""""""""""""
if __name__ == '__main__':
	MainClass()


