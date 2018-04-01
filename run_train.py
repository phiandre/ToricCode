import numpy as np
from RL import RLsys
from Env import Env
from GenerateToricData import Generate
import time
import os.path
import pickle


class MainClass:
	
	
	def __init__(self):
		# creates a new filename each time we run the code
		tmp = list('numSteps1.npy')
		static_element = 1
		while os.path.isfile("".join(tmp)):
			static_element += 1
			tmp[8] = str(static_element)
		self.filename="".join(tmp)
		
		self.run()
        
	def run(self):
		flip = np.arange(5)
		size=5
		actions=4

		rl = RLsys(4, size)
		
		humRep=np.load('ToricCodeHuman.npy')
		comRep=np.load('ToricCodeComputer.npy')
		print(comRep[:,:,3])
		
		iterations = np.zeros(comRep.shape[2])
		
		for i in range(comRep.shape[2]):
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
		print("Saving data in " + self.filename)
		np.save(self.filename,iterations)
		with open('trained_network.pkl', 'wb') as output:
			rl.qnet.network.save('trainedNetwork.h5')
		
        
       
"""""""""""""""""""""""""""""""""""""""""""""
Mainmetod, här körs själva simuleringen.
"""""""""""""""""""""""""""""""""""""""""""""
if __name__ == '__main__':
	MainClass()
	
				