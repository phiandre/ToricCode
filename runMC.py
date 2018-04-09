import numpy as np
from RLMC import RLsys
from Env import Env
from GenerateToricData import Generate
import time
import os.path
from keras.models import load_model


class MainClass:
	
	def __init__(self):
		# creates a new filename each time we run the code
		self.X = 0
		self.n = 0

		self.networkName = 'trainedNetwork.h5'

		self.gamma = 1
		tmp = list('numSteps1.npy')
		static_element = 1
		while os.path.isfile("".join(tmp)):
			static_element += 1
			tmp[8] = str(static_element)
		self.filename="".join(tmp)
		
		self.run()
        
	def run(self):

		importNetwork = load_model(self.networkName)

		rl = RLsys(4, importNetwork.input_shape[2])
		rl.qnet.network = importNetwork

		humRep=np.load('ToricCodeHuman.npy')
		comRep=np.load('ToricCodeComputer.npy')
		iterations = np.zeros(comRep.shape[2])
		
		for i in range(comRep.shape[2]):
			state=comRep[:,:,i]
			humanRep = humRep[:,:,i]
			env = Env(state, humanRep, checkGroundState=True)
			numIter = 0
			
			#######################################
			# Här utförs Monte Carlo för varje fall
			#######################################
			while len(env.getErrors()) > 0:

				numIter = numIter + 1
				C = 1
				R = 0
				firstA = -1
				"""""""""""""""""""""""""""""""""""""""
				Skapa ett nytt temporär Env-objekt och
				initiera det till att vara just där
				fallet står just nu.
				"""""""""""""""""""""""""""""""""""""""
				tempState = env.state
				tempRep = env.humanState
				tempEnv = Env(tempState, tempRep, groundState=env.groundState, checkGroundState=True)
				#######################################
				# Här utförs Monte Carlo för varje steg
				#######################################
				while len(tempEnv.getErrors()) > 0:
					# Hämta nästa action som bör tas
					observation = tempEnv.getObservation()
					a, e = rl.choose_action(observation)
					# Spara första action som tas Q(s,a)
					if firstA == -1:
						firstA = a
					# Utför förflyttning och iterera reward
					r = tempEnv.moveError(a, e)
					C = C*self.gamma
					R = R + C*r

				# Lär Q(s)
				if firstA != -1:
					rl.learn(tempState, firstA, R)	

				# Ta ett nytt steg
				observation = env.getObservation()
				a, e = rl.choose_action(observation)
				r = env.moveError(a, e)
			
			print("reward taken: "+str(r))
			print("Steps taken at iteration: " +str(i) + ": ", numIter)
			if r == 100:
				self.X += 1
			self.n += 1
			print("Average correct GS: "+str(self.X/self.n))
			iterations[i] = numIter

		print("Saving data in " + self.filename)
		np.save(self.filename,iterations)
		
		rl.qnet.network.save('MCtrainedNetwork1.h5')
		
		"""""
		En början på implemantation av Monte-Carlo learning:
		Behöver ändra i reward-funktionen eller på något annat sätt ta hänsyn till sigmaoperatorerna 
		(vilket grundtillstånd vi kommer till)
		
			env = Env(state)
			observation = env.getObservation
			C = 1
			Q = 0
			
			while observation != 'terminal':
				a, e = rl.choose_action(observation)
				reward = env.moveError(a, e)
				Q = Q+C*reward
				C = C*self.gamma
				observation = env.getObservation
				
				
			Q = Q+C*reward	
				
			# Update the neural network
			self.qnet.improveQ(state, Q)
		"""""
        
       
       
"""""""""""""""""""""""""""""""""""""""""""""
Mainmetod, här körs själva simuleringen.
"""""""""""""""""""""""""""""""""""""""""""""
if __name__ == '__main__':
	MainClass()
	
				
