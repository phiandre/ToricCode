"""""""""""""""""""""""""""""""""""""""""""""""""""

Klass för att utföra Monte-Carlo iterationer.

Skapar ett RLMC-objekt, som i sin tur har ett
QNet-objekt. Nätverket initieras till nätverket
som är sparat i strängen networkName. Använder TD-
lambda.

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
		self.gamma = 0.8
		# Variabler för statistik
		self.X = 0
		self.n = 0
		self.lamda = 10
		self.epsilon = 0.9
		# Filnamn för det tränade nätverket
		self.networkName = 'trainedNetwork15.h5'
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
		rl = RLsys(4, importNetwork.input_shape[2], e_greedy=self.epsilon)
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
			endReward = 0
			while len(env.errors) > 0:

				observation = env.getObservation()
				action, error = rl.choose_action(observation)
				# Ta ut tillstånd som flyttades
				state = observation[:,:,error]
				# Skapa en kopia av Env-objektet
				tempEnv = env.copy()
				###############################################
				# Utför nästkommande steg UTAN epsilon-greedy #
				###############################################	
				# Genom att sätta epsilon till ett så undviker epsilon-greedy.
				rl.epsilon = 1
				# Återställ reward och lambda för uppskattning
				reward = 0
				L = 0
				#####################################################
				# Utför här en policy-walk för att uppskatta Q(s,a) #
				#####################################################		
				while len(tempEnv.errors) > 0 and L < self.lamda:
					# Första iterationen medför samma observation som ovan
					obs = tempEnv.getObservation()
					a, e = rl.choose_action(obs)
					# Beräkna reward för lärandet av nätverk
					newReward = tempEnv.moveError(a, e)
					reward = newReward + self.gamma * reward
					L += 1
					if L == self.lamda and len(tempEnv.errors) > 0:
						obs = tempEnv.getObservation()
						reward = np.amax(rl.predQ(obs)) + self.gamma * reward

				rl.learn(state, action, reward)
				##############################################
				# Utför nästkommande steg MED epsilon-greedy #
				##############################################
				# Sätt epsilon tillbaka till ursprunliga värdet
				rl.epsilon = self.epsilon
				#########################################################
				# Nu är nätverket uppdaterat så vi uppskattar Q på nytt #
				#########################################################
				newAction, newError = rl.choose_action(observation)
				endReward = env.moveError(newAction, newError)

				currIterations += 1

			# Uppdatera iterations
			iterations[i] = currIterations
			#####################
			# Skriv ut resultat #
			#####################
			print("Reward vid sista steg: "+str(endReward))
			print("Antal iterationer: "+str(currIterations))

		# Spara data (fungerar ej nu)
		print("Saving data in " + self.filename)
		np.save(self.filename,iterations)

		# Spara data (fungerar ej nu)
		print("Saving data in " + self.filename)
		np.save(self.filename,iterations)


       
"""""""""""""""""""""""""""""""""""""""""""""
Mainmetod, här körs själva simuleringen.
"""""""""""""""""""""""""""""""""""""""""""""
if __name__ == '__main__':
	MainClass()
	
				
