import numpy as np
from RL import RLsys
from RL_actor import RLsys as RL_act
from Env import Env
from GenerateToricData import Generate
from keras.models import load_model
import time
import os.path
import pickle
import math
from Blossom import Blossom 
import time
from QNet import QNet

class MainClass
	def __init__(self):
		# Alla booleans
		self.loadNetwork = False #train an existing network
		self.gsRGrowth = np.load("Tweaks/GSgrowth.npy")
		
		self.checkGS = np.load("Tweaks/checkGS.npy")
		
		#Epsilon decay parameters
		
		self.epsilonDecay = np.load("Tweaks/epsilonDecay.npy")
		if self.epsilonDecay:
			self.alpha = np.load("Tweaks/alpha.npy") 		# flyttar "änden" på epsilon-kurvan
			self.k = np.load("Tweaks/k.npy")			# flyttar "mitten" på epsilon-kurvan
		
		self.comRep = np.load('ToricCodeComputer.npy')
		self.humRep=np.load('ToricCodeHuman.npy')
		
		self.networkName = 'trainedNetwork42.h5' 
		
		self.saveRate = 100 #how often the network is saved

		# creates a new filename for numSteps each time we run the code
		self.getFilename()
		self.size = self.comrep.shape[0]
		self.avgTol = 1000 # Den mängd datapunkter som average tas över
		self.fR = np.load("Tweaks/correctGsR.npy") # asymptotic Ground State reward
		self.stoppVillkor = True
		
		self.globalQnet = QNet(self.size)
		if self.loadNetwork:
			importNetwork = load_model(self.networkName)
			self.GlobalQnet.network = importNetwork
		
		
		self.run()
		
		
		
		# En actor rör sig i enviromenten och samlar in data till minnesbanken
	def actor(self)
		actorRL = RL_act(4,size)
		actorRL.qnet = self.GlobalQnet
		
		humRep = self.humRep
		comRep = self.comRep
		
		# Antalet träningsfall
		n = humrep.shape[2]
		incorrectGsR = np.load("Tweaks/incorrectGsR.npy")
		stepR = np.load("Tweaks/stepR.npy")
		numSteps = 0
		if self.gsRGrowth:
			A = np.load("Tweaks/AGS.npy")
			B = np.load("Tweaks/BGS.npy")
			w = np.load("Tweaks/wGS.npy")
			b = np.load("Tweaks/bGS.npy")
		
		while self.stoppVillkor:
			i = randint(0,n)
			state = np.copy(comRep[:,:,i])
			humanRep = humRep[:,:,i]
			
			env = Env(state, humanRep, checkGroundState=self.checkGS)
			env.incorrectGsR = incorrectGsR
			env.stepR = stepR
			
			if self.epsilonDecay:
				rl.epsilon = ((self.k+trainingIteration+12000)/self.k)**(self.alpha)
			if self.gsRGrowth:
				env.correctGsR = A*np.tanh(w*(trainingIteration+b)) + B
			else:
				env.correctGsR = self.fR
			r = 0
			
			while len(env.getErrors()) > 0:
				numSteps += 1
				observation = env.getObservation ()
				a, e = rl.choose_action(observation)
				r = env.moveError(a, e)
				new_observation = env.getObservation()


				rl.storeTransition(observation[:,:,e], a, r, new_observation)
		
		
		
		
		
	def rotateHumanRep(self,humanRep,j):
		tmp = np.concatenate([humanRep, humanRep[:,0:1]],axis=1)
		tmp1 = np.concatenate([tmp,tmp[0:1,:]])
		humanRep = np.rot90(tmp1,j)
		state = humanRep[0:(humanRep.shape[0]-1),0:(humanRep.shape[1]-1)]
		return state
		
		
	def run(self)
		
		
		
		while self.stoppVillkor:
			self.stoppVillkor = # Något logiskt evaluerbart påstående som är sant tills vi är klara
		
	
	
"""""""""""""""""""""""""""""""""""""""""""""
Mainmetod, här körs själva simuleringen.
"""""""""""""""""""""""""""""""""""""""""""""
if __name__ == '__main__':
	MainClass()