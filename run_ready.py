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
		self.networkName = 'Networks/trainedNetwork25.h5'
		self.maxNumberOfIterations = 10000

		# creates a new filename each time we run the code
		self.getFilename()
		# Skriv in den belöning du ger för att komma till rätt grund tillstånd
		# påverkar ej belöning i env, men används till att räkna ut average Ground State
		
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

		rl = RLsys(4, importNetwork.input_shape[2])
		rl.qnet.network = importNetwork
		
		rl.changeEpsilon(0)
		humRep=np.load('ToricCodeHumanTest.npy')
		comRep=np.load('ToricCodeComputerTest.npy')
		
		averager = np.zeros(comRep.shape[2]*4) # Används till att räkna ut hur sannolikt algoritmen återvänder till rätt grundtillstånd
		
		n = 0
		
		print(comRep[:,:,3])
		#np.random.shuffle(comRep)
		iterations = np.zeros(comRep.shape[2])
		for i in range(min(comRep.shape[2],self.maxNumberOfIterations)):
			state=comRep[:,:,i]
			human=humRep[:,:,i]
			env = Env(state,human,checkGroundState=True)
			steps = 0
			while len(env.getErrors()) > 0:
				#print('Bana nummer ' + str(i))
				if self.graphix:
					self.printState(env)
				steps = steps + 1
				observation = env.getObservation()
				#self.printQ(observation, rl)
					
				a, e = rl.choose_action(observation)
				r = env.moveError(a, e)
				new_observation = env.getObservation()
				
			if r == env.cGS:
				averager[n] = 1
			n += 1
				
			average = np.sum(averager)/n
			
			print("Steps taken at iteration " +str(i) + ": ", steps)
			print("Probability of correct GS so far: " + str(average*100) + " %")
			print(" ")

		print("Correct GS in " + str(sum(averager)) + " of " + str(n) + " cases")
		print("That is " + str(average*100) + " %")
		
		print("Saving data in " + self.filename)
		np.save(self.filename,iterations)

        
       
       
"""""""""""""""""""""""""""""""""""""""""""""
Mainmetod, här körs själva simuleringen.
"""""""""""""""""""""""""""""""""""""""""""""
if __name__ == '__main__':
	MainClass()


