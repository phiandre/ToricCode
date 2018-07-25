import numpy as np
from RL import RLsys
from RL_actor import RLsys as RL_act
from RL_learner import RLsys as RL_learn
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
import pathos
import multiprocess as mp
import random

def adHocProcess1(n):
	print('Starting demo process 1!')
	time.sleep(n[0])
	print('Ending demo process 1!')
def adHocProcess2(n):
	print('Starting demo process 2!')
	time.sleep(n[0])
	print('Ending demo process 2!')
	
class Bundle:
	def __init__(self,weights,memory,gradientStorage):
		self.memory = memory
		self.comRep = np.load('ToricCodeComputer.npy')
		self.humRep=np.load('ToricCodeHuman.npy')
		self.gradientStorage = gradientStorage
		self.miniBatchSize = 64
		self.stoppVillkor = True
		self.size = self.comRep.shape[0]
		self.weights = weights
		
		#self.checkGS = np.load("Tweaks/checkGS.npy")
	def actor(self):
		print('Initializing actor!')
		actorRL = RL_act(4,self.size)
		actorRL.qnet.network.set_weights(self.weights)
		
		humRep = self.humRep
		comRep = self.comRep
		
		# Antalet träningsfall
		n = humRep.shape[2]
		"""
		incorrectGsR = np.load("Tweaks/incorrectGsR.npy")
		stepR = np.load("Tweaks/stepR.npy")
		"""
		stepR = -1
		numSteps = 0
		
		
		actorRL.memory = self.memory
		"""
		if self.gsRGrowth:
			A = np.load("Tweaks/AGS.npy")
			B = np.load("Tweaks/BGS.npy")
			w = np.load("Tweaks/wGS.npy")
			b = np.load("Tweaks/bGS.npy")
		"""
		while self.stoppVillkor:
			i = random.randint(0,n)
			state = np.copy(comRep[:,:,i])
			humanRep = humRep[:,:,i]
			
			env = Env(state, humanRep, checkGroundState=False)
			#env.incorrectGsR = incorrectGsR
			env.stepR = stepR
			"""
			if self.epsilonDecay:
				actorRL.epsilon = ((self.k+trainingIteration+12000)/self.k)**(self.alpha)
			if self.gsRGrowth:
				env.correctGsR = A*np.tanh(w*(trainingIteration+b)) + B
			else:
				env.correctGsR = self.fR
			"""
			r = 0
			
			while len(env.getErrors()) > 0:
				numSteps += 1
				observation = env.getObservation ()
				a, e = actorRL.choose_action(observation)
				r = env.moveError(a, e)
				new_observation = env.getObservation()


				actorRL.storeTransition(observation[:,:,e], a, r, new_observation)
				print("Memory according to actor:\n",len(self.memory))
				

	def learner(self):
		print('Initializing learner!')
		rl = RL_learn(4,self.size)
		rl.qnet.network.set_weights(self.weights)
		while self.stoppVillkor:
			batch = []
			print("Memory according to LEARNER:\n",len(self.memory))
			if len(self.memory) > 0:
				for i in range(self.miniBatchSize):
					j = random.randint(0,(len(self.memory)-1))
					batch.append(self.memory[j])
				gradients = rl.gradPrep(batch)
				print("length of gradients:")
				print(len(gradients[1]))
				self.gradientStorage.append(gradients)
				#print(self.gradientStorage)
				
	

				
class MainClass:
	def __init__(self):
		# Alla booleans
		self.loadNetwork = False #train an existing network
		self.gsRGrowth = np.load("Tweaks/GSgrowth.npy")
		self.start = time.time()
		
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
		self.miniBatchSize = 32
		# creates a new filename for numSteps each time we run the code
		self.size = self.comRep.shape[0]
		self.avgTol = 1000 # Den mängd datapunkter som average tas över
		self.fR = np.load("Tweaks/correctGsR.npy") # asymptotic Ground State reward
		self.stoppVillkor = True
		
		self.globalQnet = QNet(self.size)
		self.gradientStorage = []
		
		if self.loadNetwork:
			importNetwork = load_model(self.networkName)
			self.GlobalQnet.network = importNetwork
		
		self.memoryList = []
		self.run()
		
		
		
		# En actor rör sig i enviromenten och samlar in data till minnesbanken
	
	
	
	def parameterServer(self, weights, gradientStorage):
		
		print(self.gradientStorage)
	
	def demoprocess1(self,n):
		print('Starting demo process 1!')
		sleep(n)
		print('Ending demo process 1!')
	
	def demoprocess2(self,n):
		print('Starting demo process 2!')
		sleep(n)
		print('Ending demo process 2!')
		
	def rotateHumanRep(self,humanRep,j):
		tmp = np.concatenate([humanRep, humanRep[:,0:1]],axis=1)
		tmp1 = np.concatenate([tmp,tmp[0:1,:]])
		humanRep = np.rot90(tmp1,j)
		state = humanRep[0:(humanRep.shape[0]-1),0:(humanRep.shape[1]-1)]
		return state
	
	
	def __call__(self):
		actor()
	
	def run(self):
		manager = mp.Manager()
		weights = self.globalQnet.network.get_weights()
		memory = manager.list()
		gradientStorage = manager.list()
		bundle = Bundle(weights,memory,gradientStorage)
		
		print('About to initialize!')
		act1 = mp.Process(target = bundle.actor)
		print('Something should be initialized!')
		learn1 = mp.Process(target = bundle.learner)
		print('Everything should be initialized!')
		
		act1.start()
		learn1.start()
		
		
		
		""" Följande Ad Hoc-process är testad och funkar!
		p1 = mp.Process(target = adHocProcess1, args=([1],))
		p2 = mp.Process(target = adHocProcess2, args=([2],))
		p1.start()
		p2.start()
		"""
		
		while self.stoppVillkor:
			self.stoppVillkor = (time.time()-self.start < 100)
		
	
	
"""""""""""""""""""""""""""""""""""""""""""""
Mainmetod, här körs själva simuleringen.
"""""""""""""""""""""""""""""""""""""""""""""
if __name__ == '__main__':
	MainClass()