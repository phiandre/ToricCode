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

########################
# Klass för programmet #
########################
class MainClass:
	
	"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
	Main-klassens konstruktor.
	"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
	def __init__(self):
		# Inlärningsgamma
		self.gamma = 1
		# Variabler för statistik
		self.X = 0
		self.n = 0
		# Filnamn för det tränade nätverket
		self.networkName = 'trainedNetwork.h5'
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
		self.rl = RLsys(4, importNetwork.input_shape[2])
		self.rl.qnet.network = importNetwork
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
			self.env = Env(state, humanRep, checkGroundState=True)
			######################################################
			# Leta efter och ta bort fel tills alla fel är borta #
			######################################################
			self.learnStep(state)
		# Spara data (fungerar ej nu)
		print("Saving data in " + self.filename)
		np.save(self.filename,iterations)

	"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
	Rekursiv metod för Monte-Carlo, nätverket uppdateras rekursivt
	genom att rl söker sig fram till ett errorlöst state.
		@param
			state: nuvarande tillstånd.
		@return
			int: reward som dyker upp efter state.
	"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""		
	def learnStep(self, state):
		# Avsluta om felen (errors) är borttagna
		if len(self.env.errors) == 0:
			return 0
		# Kopiera tillstånd för att slippa fel i Env
		copiedState = np.copy(state)
		# Bestäm action, error och tillhörande reward
		observation = self.env.getObservation()
		action, error = self.rl.choose_action(observation)
		# Hämta nuvarande och nästkommande reward
		newReward = self.env.moveError(action, error)
		upcomingReward = self.learnStep(copiedState)
		# Uppdatera reward (som skall uppdateras i nätverket)
		reward = newReward + self.gamma * upcomingReward
		# Uppdatera nätverket
		self.rl.learn(copiedState, action, reward)
		# Returnera reward uppnådd hittills
		return reward
       
"""""""""""""""""""""""""""""""""""""""""""""
Mainmetod, här körs själva simuleringen.
"""""""""""""""""""""""""""""""""""""""""""""
if __name__ == '__main__':
	MainClass()
	
				
