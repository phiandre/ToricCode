import numpy as np
from RL import RLsys
from Env import Env
from GenerateToricData import Generate
import time
import os.path
import pickle


class MainClass:

	def __init__(self):
		#TODO värden som skall sättas innan en körning
		self.maxNumberOfIterations = 10000

		# creates a new filename each time we run the code
		tmp = list('numSteps1.npy')
		self.static_element = 1
		while os.path.isfile("".join(tmp)):
			self.static_element += 1
			tmp[8] = str(self.static_element)
		self.filename="".join(tmp)

		self.run()
        
	def run(self):
		flip = np.arange(5)
		size=10
		actions=4

		rl = RLsys(4, size)
		
		comRep=np.load('ComputerData.npy')
		humRep=np.zeros((size*2,size*2,comRep.shape[2]))
		

		iterations = np.zeros(comRep.shape[2])

		for i in range(min(comRep.shape[2],self.maxNumberOfIterations)):
			state=comRep[:,:,i]
			env = Env(state)
			numIter = 0

			while len(env.getErrors()) > 0:
				#print('Bana nummer ' + str(i))
				#print(state)
				numIter = numIter + 1
				observation = env.getObservation()
				a, e = rl.choose_action(observation)
				r = env.moveError(a, e)
				new_observation = env.getObservation()

				rl.learn(observation[:,:,e], a, r, new_observation)

			print("Steps taken at iteration " +str(i) + ": ", numIter)
			iterations[i] = numIter

		tmp = list('trainedNetwork1.h5')
		tmp[14] = str(self.static_element)
		filename = "".join(tmp)

		print("Saving data in " + self.filename)
		np.save(self.filename,iterations)

		rl.qnet.network.save(filename)

        
       
"""""""""""""""""""""""""""""""""""""""""""""
Mainmetod, här körs själva simuleringen.
"""""""""""""""""""""""""""""""""""""""""""""
if __name__ == '__main__':
	MainClass()


