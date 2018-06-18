"""""""""""""""""""""""""""""""""""""""""""""""""""

Klass för att utföra Monte-Carlo iterationer.

Skapar ett RLMC-objekt, som i sin tur har ett
QNet-objekt. Nätverket initieras till nätverket
som är sparat i strängen networkName.

"""""""""""""""""""""""""""""""""""""""""""""""""""

#################
# Importeringar #
#################
import numpy as np
from RLMC import RLsys
from Env import Env
import time
import os.path
from keras.models import load_model
from collections import deque

########################
# Klass för programmet #
########################
class MainClass:
	
	"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
	Main-klassens konstruktor, definierar filnamn där vi skall
	importera nätverket ifrån. Definierar också avtagningsfaktorn
	gamma för Monte-Carlo.
	"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
	def __init__(self):
		# Inlärningsgamma
		self.gamma = 0.8
		# Variabler för statistik
		self.X = 0
		self.n = 0
		# Sätt innan körning
		self.alpha = -0.5
		self.saveRate = 99
		self.loadNetwork = False
		# Filnamn för det tränade nätverket
		self.networkName = 'MCtrainedNetwork20.h5'
		# Skapa ett nytt filnamn för utdata
		tmp = list('numSteps1.npy')
		self.static_element = 1
		while os.path.isfile("".join(tmp)):
			self.static_element += 1
			tmp[8] = str(self.static_element)
		self.filename="".join(tmp)
		# Kör igång Monte-Carlo algoritmen
		self.run()
		
		
	def rotateHumanRep(self,humanRep,j):
		tmp = np.concatenate([humanRep, humanRep[:,0:1]],axis=1)
		tmp1 = np.concatenate([tmp,tmp[0:1,:]])
		humanRep = np.rot90(tmp1,j)
		state = humanRep[0:(humanRep.shape[0]-1),0:(humanRep.shape[1]-1)]
		return state
		

	"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
	Main-klassens Monte-Carlo algoritm för inlärning av nätverket.
	"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
	def run(self):
		
		# Ladda in träningsdatan
		humRep=np.load('ToricCodeHuman.npy')
		comRep=np.load('ToricCodeComputer.npy')
		actions = 4
		size = comRep.shape[0]
		
		steps = np.zeros(comRep.shape[2]*4)
		
		# Instansiera ett RLMC-objekt och läs in nätverket
		rl = RLsys(actions, size)

		# Ladda in modellen (nätverket) från filnamn
		if self.loadNetwork:
			importNetwork = load_model(self.networkName)
			rl.qnet.network = importNetwork
		
		

		############################################
		# Utför Monte-Carlo för varje träningsdata #
		############################################
		for i in range(comRep.shape[2]):
			for j in range(4):
				# Läs in ett träningsfall
				state=comRep[:,:,i]
				#print("comRep: ", state)
				state = np.rot90(state,j)
				#print("comRep90: ", state)
				humanRep = humRep[:,:,i]
				humanRep = self.rotateHumanRep(humanRep,j)
				
				
				env = Env(state, humanRep, checkGroundState=True)
				numSteps = 0
				currReward = 0
				empiricalQ = 0
				# Använd queue för att få konstant tidskomplexitet
				stateList = deque()
				rewardList = deque()
				actionList = deque()
				

				######################################################
				# Leta efter och ta bort fel tills alla fel är borta #
				######################################################
				while len(env.errors) > 0:
					# Bestäm action, error och tillhörande reward
					observation = env.getObservation()
					action, error = rl.choose_action(observation)
					# Ta ut tillstånd som flyttades
					state = observation[:,:,error]
					# Uppdatera rewards
					currReward = env.moveError(action, error)
					# Uppdatera listor för inlärning senare
					stateList.append(state)
					rewardList.append(currReward)
					actionList.append(action)
					###########################
					# Uppdatera för statistik #
					###########################
					numSteps += 1

				########################################
				# Uppdatera nätverket utifrån resultat #
				########################################
				
				for k in range(numSteps):
				# Skapa reward iterativt
					empiricalQ = rewardList.pop() + self.gamma * empiricalQ
				# Uppdatera nätverket
					s = stateList.pop()
					a = actionList.pop()
				# Skicka till rl för uppdatering
					rl.learn(s, a, empiricalQ)
				
				
				
				"""
				empQ = np.zeros(numSteps)
				
				stateArray = np.asarray(stateList)
				stateArray = np.flip(stateArray,0)
				#stateArray = np.swapaxes(stateArray,0,1)
				#stateArray = np.swapaxes(stateArray,1,2)
				
				actionArray = np.asarray(actionList)
				actionArray = np.flip(actionArray,0)
				
				for k in range(numSteps):
					# Skapa reward iterativt
					empiricalQ = rewardList.pop() + self.gamma * empiricalQ
					empQ[k] = empiricalQ
					# Skicka till rl för uppdatering

				rl.learn(stateArray, actionArray, empQ)
				"""
					
				
				if currReward == 10:
					self.X += 1
				
				# Uppdatera steps
				steps[self.n] = numSteps
				self.n += 1
				#####################
				# Skriv ut resultat #
				#####################
				print("Final reward: "+str(currReward))
				print("Iteration: "+str(self.n))
				print("Average correct GS: "+str(self.X/self.n))
				print("Number of steps: "+str(numSteps))
				
				if(self.n % self.saveRate == 0):
					tmp = list('MCtrainedNetwork1.h5')
					tmp[16] = str(self.static_element)
					filename = "".join(tmp)
					#print("Saving data in " + self.filename)
					np.save(self.filename,steps[0:(i+1)])
					rl.qnet.network.save(filename)

		###############################
		# Spara data (fungerar ej nu) #
		###############################
		print("Saving data in " + self.filename)
		np.save(self.filename,steps)
       
"""""""""""""""""""""""""""""""""""""""""""""
Mainmetod, här körs själva simuleringen.
"""""""""""""""""""""""""""""""""""""""""""""
if __name__ == '__main__':
	MainClass()
	
				
