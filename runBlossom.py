import numpy as np
from Blossom import Blossom
from EBlossom import EBlossom

from BlossomEnv import Env

class runBlossom:
	
	def __init__(self):
		self.X_euclidian = 0
		self.n_euclidian = 0
		
		self.n_manhattan = 0
		self.X_manhattan = 0
		self.run()
		
	def labelState(self, s, size):
		state = s
		label = 1
		for j in range(size):
			for k in range(size):
				if state[j,k] == 1:
					state[j,k] = label
					label +=1
		return state
		
	def initialize(self, size):
		humanRepresentation = np.zeros((2*size,2*size))
		for i in range(0,2*size):
			if i%2==0:
				for j in range(0,2*size):
					if j%2==1:
						humanRepresentation[i,j] = 1
					
			else:
				for j in range(0,2*size):
					if j%2==0:
						humanRepresentation[i,j] = 1
		
		return humanRepresentation
	
	def run(self):
		comRep = np.load('ToricCodeComputerTest.npy')
		humRep=np.load('ToricCodeHumanTest.npy')
		size = comRep.shape[0]
		
		for i in range(comRep.shape[2]):
			state = np.copy(comRep[:,:,i])
			humanRep = humRep[:,:,i]
			
			
			state = self.labelState(state,state.shape[0])
			
			"""
			state = np.zeros((size,size))
			state[1,3] = 1
			state[2,3] = 2
			state[3,1] = 3
			state[3,2] = 4
			state[3,3] = 5
			state[4,2] = 6
			state[4,3] = 7
			state[4,4] = 8
			
			humanRep = self.initialize(size)
			humanRep[4,7] = -1
			humanRep[7,4] = -1
			humanRep[7,8] = -1
			humanRep[8,9] = -1
			humanRep[9,6] = -1
			"""
			Eenv = Env(state, humanRep, checkGroundState = True)
			Menv = Env(state, humanRep, checkGroundState = True)
			
			if np.count_nonzero(state) > 0:
				ManhattanBlossom = Blossom(state)
				EuclidianBlossom = EBlossom(state)
				EuclidianMWPM = EuclidianBlossom.readResult()
				ManhattanMWPM = ManhattanBlossom.readResult()
				
				
				if EuclidianMWPM != ManhattanMWPM:
					for element in EuclidianMWPM:
						error1 = element[0]
						error2 = element[1]
						EuclidianReward = Eenv.blossomCancel(error1, error2)
					
					
					for element in ManhattanMWPM:
						error1 = element[0]
						error2 = element[1]
						ManhattanReward = Menv.blossomCancel(error1, error2)
					
					
					if EuclidianReward == Eenv.correctGsR:
						self.X_euclidian += 1
						#print("Euclidian decoder correct:\n", Eenv.humanState)
					#else:
						#print("Euclidian NOT CORRECT:\n", Eenv.humanState)
					self.n_euclidian += 1
					
					if ManhattanReward == Menv.correctGsR:
						self.X_manhattan += 1
						#print("Manhattan decoder correct:\n", Menv.humanState)
					#else:
						#print("Manhattan NOT CORRECT:\n", Menv.humanState)
					self.n_manhattan += 1
					
					print("Occurrence of different matching: ", self.n_euclidian / (i+1) ) 
					print("Euclidian GS: ", self.X_euclidian / self.n_euclidian)
					print("Manhattan GS: ", self.X_manhattan / self.n_manhattan)
					
			print("Iteration: ", i)
			"""
				#if(EuclidianMWPM != ManhattanMWPM):
					#print("Current state:\n",state)
					#print("MWPM Euclidian: ", EuclidianMWPM)
					#print("MWPM Manhattan: ", ManhattanMWPM)
					self.X += 1
				self.n += 1
				
				if self.n%100 == 0:
					print("Opportunity: ", self.X / self.n)
				
				for element in MWPM:
					error1 = element[0]
					error1_coords = element[1]
					error2 = element[2]
					error2_coords = element[3]
					r = env.blossomCancel(error1, error2, error1_coords, error2_coords)
				print("r: ", r)
				
				if r==5:
					self.X += 1
				else:
					print("state:\n", state)
				self.n += 1
				
				#print("Correct GS: ", self.X / self.n)
			"""
			

if __name__ == '__main__':
	runBlossom()
