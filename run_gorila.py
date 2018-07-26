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
import tensorflow as tf

	
class Bundle:
	def __init__(self,weights,memory,gradientStorage,globalCounter,updateCheck, checkerQueue, globalStep):
		self.memory = memory
		self.comRep = np.load('ToricCodeComputer.npy')
		self.humRep=np.load('ToricCodeHuman.npy')
		self.gradientStorage = gradientStorage
		self.miniBatchSize = 8
		self.stoppVillkor = True
		self.size = self.comRep.shape[0]
		self.weights = weights
		self.globalCounter = globalCounter
		self.updateCheck = updateCheck
		self.checkerQueue = checkerQueue
		self.globalStep = globalStep
		
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
			start = time.time()
			numSteps = 0
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
			if time.time()-start > 5:
				actorRL.qnet.network.set_weights(self.weights)
				start = time.time()
				

	def learner(self):
		print('Initializing learner!')
		indexQueue = False
		rl = RL_learn(4,self.size)
		rl.qnet.network.set_weights(self.weights)
		n = 0
		while not indexQueue:
			if self.checkerQueue.value:
				self.checkerQueue.value = False
				self.updateCheck.append(False)
				identity = len(self.updateCheck)-1
				self.checkerQueue.value = True
				indexQueue = True
		
		while self.stoppVillkor:
			if len(self.memory) > 0:
				batch = []
				for i in range(self.miniBatchSize):
					j = random.randint(0,(len(self.memory)-1))
					batch.append(self.memory[j])
				gradients = rl.gradPrep(batch)
				self.gradientStorage.append(gradients)

				if n % 4 == 0:
					rl.qnet.network.set_weights(self.weights)
				if (self.globalCounter.value >= self.globalStep.value) and (not self.updateCheck[identity]):
					rl.targetNet.network.set_weights(self.weights)
					self.updateCheck[identity] = True
				n += 1
				
class ParameterServer:
	def __init__(self, weights, gradientStorage, globalCounter, updateCheck, globalStep):
		self.weights = weights
		self.gradientStorage = gradientStorage
		self.stoppVillkor = True
		self.updateCheck = updateCheck
		self.globalStep = globalStep
		self.globalCounter = globalCounter
		self.alpha = 0.001
		self.updateTimer = time.time()
		
	def networkOptimizer(self):
		print('Initializing parameter server')
		while self.stoppVillkor:
			if len(self.gradientStorage) > 4:
				
				print('Updating network...')
				print('Time between updates: ',(time.time()-self.updateTimer))
				self.updateTimer = time.time()
				averageStep = [-self.alpha*sum(x)/len(self.gradientStorage) for x in zip(*self.gradientStorage)]
				self.weights[:] = [sum(x) for x in zip(averageStep,self.weights)]
				self.gradientStorage[:]=[]
				self.globalCounter.value += 1
				print('Update count: ',self.globalCounter.value)
			if all(self.updateCheck):
				print('Target Networks are synced!')
				self.updateCheck[:] = [not i for i in self.updateCheck]
				self.globalCounter.value = 0
				

				
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
	
	
	
	
				#print(self.gradientStorage)
	

		
	def rotateHumanRep(self,humanRep,j):
		tmp = np.concatenate([humanRep, humanRep[:,0:1]],axis=1)
		tmp1 = np.concatenate([tmp,tmp[0:1,:]])
		humanRep = np.rot90(tmp1,j)
		state = humanRep[0:(humanRep.shape[0]-1),0:(humanRep.shape[1]-1)]
		return state
	
	

	
	def run(self):
		manager = mp.Manager()
		
		
		
		"""Delade saker"""
		globalWeightList = manager.list()							# Nätverkets vikter
		globalWeightList[:] = self.globalQnet.network.get_weights()
		memory = manager.list()										# Minnesbuffert
		gradientStorage = manager.list()							# Här förvaras gradienter tills nätverket uppdateras
		globalCounter = manager.Value('i',0)						# Hur många gånger nätverket uppdaterats sedan senaste Targetsynkning
		checkerQueue = manager.Value('boolean',True)				# True = ledigt för learner att få identitet
		updateCheck = manager.list()								# När index för alla learners är True så är Targetsynkning över
		globalStep = manager.Value('i',10)							# Antal uppdateringar som ska ske mellan Targetsynkningar
		
		"""Initiera instanser av parameterServer and Bundle"""
		# En bundle är en grupp actors och learners med gemensam minnesbuffert
		bundle = Bundle(self.globalQnet.network.get_weights(),memory,gradientStorage, globalCounter, updateCheck, checkerQueue, globalStep)
		# På parameterservern förvaras och uppdateras nätverkets vikter
		parameterServer = ParameterServer(globalWeightList,gradientStorage, globalCounter, updateCheck, globalStep)
		
		"""Definiera processer"""
		# Här definieras
		act1 = mp.Process(target = bundle.actor)
		learn1 = mp.Process(target = bundle.learner)
		param = mp.Process(target = parameterServer.networkOptimizer)
		
		"""Starta processer"""
		act1.start()
		learn1.start()
		param.start()
		
		
		saveStart = time.time()
		
		while self.stoppVillkor:
			
			if (time.time() - saveStart) > self.saveRate:
				print('Saving network')
				self.globalQnet.network.set_weights(globalWeightList)
				self.globalQnet.network.save('Networks/GorilaNetwork.h5')
				saveStart = time.time()
			self.stoppVillkor = True
		
		"""run() får ej avslutas före processerna, då run() har alla delade saker i sig"""
		act1.join()
		learn1.join()
		param.join()
	
	
"""""""""""""""""""""""""""""""""""""""""""""
Mainmetod, här körs själva simuleringen.
"""""""""""""""""""""""""""""""""""""""""""""
if __name__ == '__main__':
	MainClass()