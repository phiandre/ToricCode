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
from GenerateToricData import Generate
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
		self.gamma = 1
		# Variabler för statistik
		self.X = 0
		self.n = 0
		# Filnamn för det tränade nätverket
		self.networkName = 'trainedNetwork14.h5'
		# Skapa ett nytt filnamn för utdata
		tmp = list('numSteps1.npy')
		static_element = 1
		while os.path.isfile("".join(tmp)):
			static_element += 1
			tmp[8] = str(static_element)
		self.filename="".join(tmp)
		# Kör igång Monte-Carlo algoritmen
		self.run()

	"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
	Main-klassens Monte-Carlo algoritm för inlärning av nätverket.
	"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
	def run(self):

		# Ladda in modellen (nätverket) från filnamn
		importNetwork = load_model(self.networkName)
		# Instansiera ett RLMC-objekt och läs in nätverket
		rl = RLsys(4, importNetwork.input_shape[2])
		rl.qnet.network = importNetwork
		# Ladda in träningsdatan
		humRep=np.load('ToricCodeHuman.npy')
		comRep=np.load('ToricCodeComputer.npy')
		iterations = np.zeros(comRep.shape[2])

		############################################
		# Utför Monte-Carlo för varje träningsdata #
		############################################
		for i in range(comRep.shape[2]):
			# Läs in ett träningsfall
			state=comRep[:,:,i]
			humanRep = humRep[:,:,i]
			env = Env(state, humanRep, checkGroundState=True)
			currIterations = 0
			currReward = 0
			reward = 0
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
				currIterations += 1

			########################################
			# Uppdatera nätverket utifrån resultat #
			########################################
			for i in range(currIterations):
				# Skapa reward iterativt
				reward = rewardList.pop() + self.gamma * reward
				# Uppdatera nätverket
				rl.learn(stateList.pop(), actionList.pop(), reward)

			# Uppdatera iterations
			iterations[i] = currIterations
			#####################
			# Skriv ut resultat #
			#####################
			print("Reward vid sista steg: "+str(currReward))
			print("Antal iterationer: "+str(currIterations))

		# Spara data (fungerar ej nu)
		print("Saving data in " + self.filename)
		np.save(self.filename,iterations)
       
"""""""""""""""""""""""""""""""""""""""""""""
Mainmetod, här körs själva simuleringen.
"""""""""""""""""""""""""""""""""""""""""""""
if __name__ == '__main__':
	MainClass()
	
				
