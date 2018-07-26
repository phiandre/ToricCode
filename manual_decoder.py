"""""""""""""""""""""
	Importeringar
"""""""""""""""""""""
import numpy as np
from Env import Env
from Blossom import Blossom
from GenerateToricData import Generate
import time
import os.path
import pickle
import os

"""""""""""""""""""""""
	Klassdefinition
"""""""""""""""""""""""
class MainClass:

	def __init__(self):
		#TODO värden som skall sättas innan varje körning
		self.graphix = False
		self.saveData = False
		self.networkName = 'Networks/trainedNetwork32.h5'
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
		
	def cls(self):
		os.system('cls' if os.name == 'nt' else 'clear')
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

		
		bCorr = 0
		largeNum = 0
		humRep=np.load('ToricCodeHumanTest.npy')
		comRep=np.load('ToricCodeComputerTest.npy')
		print(comRep[:,:,3])
		#np.random.shuffle(comRep)
		iterations = np.zeros(comRep.shape[2])
		for i in range(min(comRep.shape[2],self.maxNumberOfIterations)):
			dist = 0
			state=comRep[:,:,i]
			human=humRep[:,:,i]

			if np.count_nonzero(state) > 0:
				state_ =np.copy(state)
				state_ = self.labelState(state_,size)


			env = Env(state,human,checkGroundState=True)
			numIter = 0
			while len(env.getErrors()) > 0:
				invalid = False
				#print('Bana nummer ' + str(i))
				if self.graphix:
					self.printState(env)
				numIter = numIter + 1
				#self.printQ(observation, rl)
				#print(env.state)
				observation = env.getObservation()
				move = True
				while move:
					
					staty = np.copy(env.state)
					state_l = self.labelState(staty,size)
					
					err = True
					while err:
						self.cls()
						
						print("Stage " + str(self.n+1) + ":\n", state_l)
						if invalid:
							print("Invalid response!")
						e = input("Which of the " + str(len(env.getErrors())) + " errors do you want to move?\n")
						if e.isdigit():
							e = int(e)
							if e >= 1 and e <= len(env.getErrors()):
								err = False
							else:
								invalid = True
						elif e == 'exit':
							self.cls()
							exit()
						else:
							invalid = True
					
					invalid = False
					e -= 1
					move_ = True
					while move_:
						self.cls()
						print("Perspective of error " + str(e)+":\n",observation[:,:,(e)])
						if invalid:
							print("Invalid response!")
						act = input("up/down/left/right/back; w/a/s/d/b \n")
						if (act == 'up') or (act == 'w'):
							a = 0
							r = env.moveError(a,e)
							move = False
							move_ = False
						elif (act == 'down') or (act == 's'):
							a = 1
							r = env.moveError(a,e)
							move = False
							move_ = False
						elif (act == 'left') or (act == 'a'):
							a = 2
							r = env.moveError(a,e)
							move = False
							move_ = False
						elif (act == 'right') or (act == 'd'):
							a = 3
							r = env.moveError(a,e)
							move = False
							move_ = False
						elif (act == 'back') or (act == 'b'):
							
							move_ = False
						elif (act == 'exit'):
							self.cls()
							exit()
						else:
							invalid = True
				
			
			self.cls()
			print("Level finished!")
			print("You took " + str(numIter) + " steps")
			print("This is the current error path:\n",env.humanState)


			if r == env.correctGsR:
				self.X += 1
				print("Topological Ground State is correct!")
			else:
				print("Topological Ground State is incorrect!")



			self.n += 1
			print("So far, your success rate is: " + str(100*(self.X/self.n)) + " %")
			nothing = input("enter 'exit' to exit, or anything else to proceed\n")
			if nothing == 'exit':
				self.cls()
				exit()
			"""
			print("MWPM correct ", bCorr / self.n)
			"""
			iterations[i] = numIter




		print("Saving data in " + self.filename)
		np.save(self.filename,iterations)

        
       
       
"""""""""""""""""""""""""""""""""""""""""""""
Mainmetod, här körs själva simuleringen.
"""""""""""""""""""""""""""""""""""""""""""""
if __name__ == '__main__':
	MainClass()


