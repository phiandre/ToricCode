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
		self.saveData = False
		self.maxNumberOfIterations = 5000
		self.alpha = -0.5
		self.saverate = 1000

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
			self.filename = 'numSteps1.npy'
			self.static_element = 1

			while os.path.isfile("".join(tmp)):
				self.static_element += 1
				tmp[8] = str(self.static_element)
				self.filename = "".join(tmp)




	def run(self):
		flip = np.arange(5)
		size=9
		actions=4

		rl = RLsys(4, size)
		
		comRep=np.load('ToricCodeComputer.npy')
		humRep=np.zeros((size*2,size*2,comRep.shape[2]))

		iterations = np.zeros(comRep.shape[2])

		for i in range(min(comRep.shape[2],self.maxNumberOfIterations)):
			state=comRep[:,:,i]
			env = Env(state)
			numIter = 0
			rl.epsilon = (1+x)**(self.alpha)

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

			if(i % self.saverate == 0):
				if(self.saveData):

					tmp = list('trainedNetwork1.h5')
					tmp[14] = str(self.static_element)
					filename = "/Users/nikfor/Desktop/Kandidat/Saves/" + "".join(tmp)

					#print("Saving data in " + self.filename)
					np.save(self.filename,iterations[0:(i+1)])
					np.savetxt('test.txt',iterations[0:(i+1)])

					rl.qnet.network.save(filename)

				else:
					tmp = list('trainedNetwork1.h5')
					tmp[14] = str(self.static_element)
					filename = "".join(tmp)

					#print("Saving data in " + self.filename)
					np.save(self.filename,iterations[0:(i+1)])
					np.savetxt('test.txt',iterations[0:(i+1)])

					rl.qnet.network.save(filename)


		

"""""""""""""""""""""""""""""""""""""""""""""
Mainmetod, här körs själva simuleringen.
"""""""""""""""""""""""""""""""""""""""""""""
if __name__ == '__main__':
	MainClass()


