import numpy as np
from RL import RLsys
from Env import Env
from RL2 import RLsys as RLsys2
from GenerateToricData import Generate
from keras.models import load_model
import time
import os.path
import pickle


class MainClass:

	def __init__(self):
		
		self.alpha = 0.18 #epsilon decay
		
		self.loadNetwork = False #train an existing network
		self.RL1networkName = 'Networks/trainedNetwork15.h5'
		self.networkName = 'Networks/89trainedNetwork47.h5'
		
		self.saveRate = 99 #how often the network is saved

		# creates a new filename for numSteps each time we run the code
		self.getFilename()
		self.X=0
		self.n=0
		self.avgTol = 1000
		
		self.GSr = 5
		
		self.veriComRep = np.load('ToricCodeComputerTest.npy')
		self.veriHumRep = np.load('ToricCodeHumanTest.npy')
		self.size = self.veriComRep.shape[0]
		self.baiter = 20
		self.avgHigh = 0.9
		self.avgLow = 0.86
		self.goal = 0.9
		
		
		
		self.run()
		


	def getFilename(self):
		tmp = list('Steps/numSteps1.npy')
		self.filename = "Steps/"+'numSteps1.npy'
		self.static_element = 1

		while os.path.isfile("".join(tmp)):
			self.static_element += 1
			tmp[14] = str(self.static_element)
			self.filename = "".join(tmp)

	def rotateHumanRep(self,humanRep,j):
		tmp = np.concatenate([humanRep, humanRep[:,0:1]],axis=1)
		tmp1 = np.concatenate([tmp,tmp[0:1,:]])
		humanRep = np.rot90(tmp1,j)
		state = humanRep[0:(humanRep.shape[0]-1),0:(humanRep.shape[1]-1)]
		return state
	
	def verify(self,rl):
		interval = 200
		steps = np.zeros(interval)
		averager = np.zeros(interval)
		trainingIteration = 0
		maxSteps = 300
		
		for i in range(interval):
			state = self.veriComRep[:,:,i]
			humanRep = self.veriHumRep[:,:,i]
			env = Env(state, humanRep, checkGroundState=True)
			
			numSteps = 0
			
			
			while len(env.getErrors()) > 0:
					numSteps = numSteps + 1
					observation = env.getObservation()
					observation2 = env.getObservation2()
					a, e = rl.choose_action(observation, observation2)
					r = env.moveError(a, e)
					if numSteps >= maxSteps:
						break
			if r == self.GSr:
				averager[trainingIteration] = 1
			trainingIteration += 1
		average = (np.sum(averager))/(interval)
		return average

	def run(self):
		actions = 4
		comRep = np.load('ToricCodeComputer.npy')
		humRep = np.load('ToricCodeHuman.npy')
		
		rl2 = RLsys(actions, self.size)
		rl = RLsys2(actions, self.size)
		
		
		if self.loadNetwork:
			importNetwork = load_model(self.networkName)
			rl2.qnet.network = importNetwork
		
		importNetwork = load_model(self.RL1networkName)
		rl.qnet.network = importNetwork
		
		steps = np.zeros(comRep.shape[2]*4)
		averager = np.zeros(self.avgTol)
		
		bait = 1
		
		trainingIteration = 0
		k = bait
		for i in range(comRep.shape[2]):
			for j in range(4):
				state = comRep[:,:,i]
				
				state = np.rot90(state,j)
				
				humanRep = humRep[:,:,i]
				humanRep = self.rotateHumanRep(humanRep,j)
				
				env = Env(state, humanRep, checkGroundState=True)
				numSteps = 0
				if self.loadNetwork:
					rl.epsilon = 0.2 #(8000+1+trainingIteration)**(self.alpha)
				else:
					rl.epsilon = (1+trainingIteration)**(self.alpha)
				while len(env.getErrors()) > 0:
					numSteps = numSteps + 1
					observation = env.getObservation()
					observation2 = env.getObservation2()
					a, e = rl.choose_action(observation)
					r = env.moveError(a, e)
					new_observation = env.getObservation()
					new_observation2 = env.getObservation2()
					
					rl2.learn(observation[:,:,e], observation2[:,:,e], a, r, new_observation, new_observation2)
				self.n += 1
				if (self.n) <= self.avgTol:
					if r == self.GSr:
						averager[trainingIteration] = 1
					average = (np.sum(averager))/(self.n)
				else:
					if r == self.GSr:
						avg1=averager[1:]
						avg2=np.ones((1))
						
						averager = np.concatenate((avg1, avg2))
					else:
						avg1 = averager[1:]
						avg2 = np.zeros((1))
						averager = np.concatenate((avg1, avg2))
					average = (np.sum(averager))/self.avgTol
				
				steps[trainingIteration] = numSteps
				
				
				
				
				print(' ')
				print('Episode ' + str(trainingIteration))
				print("Steps taken: "+ str(numSteps))
				if r == self.GSr:
					print("Groundstate is RIGHT!")
				else:
					print("Groundstate is WRONG!")
				print("Probability of correct GS last " + str(min(self.avgTol,self.n))+ " episodes: " + str(100*average) + " %")

				if(trainingIteration % self.saveRate == 0):
					tmp = list('Networks/trainedNetwork1.h5')
					tmp[23] = str(self.static_element)
					filename = "".join(tmp)
					
					filename = "".join(tmp)	

					np.save(self.filename,steps[0:(trainingIteration+1)])

					rl2.qnet.network.save(filename)
					print("Network saved")
				"""
				
				if self.loadNetwork:
					if(average > self.avgHigh ):
						if k % bait == 0:
							veriAv = self.verify(rl)
							if veriAv < self.avgLow:
								bait = self.baiter
							else:
								bait = 1
							print("veriAv = " + str(veriAv))
							if veriAv > self.avgHigh:
								tmp = list('Networks/88trainedNetwork1.h5')
								tmp[25] = str(self.static_element)
								filename = "".join(tmp)	

								np.save(self.filename,steps[0:(trainingIteration+1)])

								rl.qnet.network.save(filename)
								
								tmp = list('Networks/89trainedNetwork1.h5')
								tmp[25] = str(self.static_element)
								filename = "".join(tmp)	
							

								rl.qnet.network.save(filename)
								print("Decent network - saving")
								if veriAv > self.goal:
									print("Goal rached!")
									quit()
						k += 1
						
					if(average < self.avgHigh ):
						k = bait
				else:
					if self.n > self.avgTol:
						if(average > self.avgHigh ):
							
							if k % bait == 0:
								veriAv = self.verify(rl)
								if veriAv < self.avgLow:
									bait = self.baiter
								else:
									bait = 1
								print("veriAv = " + str(veriAv))
								if veriAv > self.avgHigh:
									tmp = list('Networks/88trainedNetwork1.h5')
									tmp[25] = str(self.static_element)
									filename = "".join(tmp)	

									np.save(self.filename,steps[0:(trainingIteration+1)])

									rl.qnet.network.save(filename)
									
									tmp = list('Networks/89trainedNetwork1.h5')
									tmp[25] = str(self.static_element)
									filename = "".join(tmp)	
									

									rl.qnet.network.save(filename)
									print("Decent network - saving")
									if veriAv > self.goal:
										print("Goal rached!")
										quit()
									
							k += 1
						if(average < self.avgHigh ):
							k = bait
				"""
				trainingIteration = trainingIteration + 1


		

"""""""""""""""""""""""""""""""""""""""""""""
Mainmetod, här körs själva simuleringen.
"""""""""""""""""""""""""""""""""""""""""""""
if __name__ == '__main__':
	MainClass()


