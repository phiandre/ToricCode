import numpy as np
from Blossom import Blossom
from EBlossom import EBlossom
from BlossomEnv import Env
from MCGraph import MCGraph
from itertools import groupby

class runBlossom:
	
	def __init__(self):
		self.X = 0
		self.n = 0
		
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
		
		plaqCount = 0
		
		plaq = np.zeros(12)
		plaqGS = np.zeros(12)
		
		for i in range(comRep.shape[2]):
			state = np.copy(comRep[:,:,i])
			humanRep = humRep[:,:,i]
			
			
			state = self.labelState(state,state.shape[0])
			
			
			
			env = Env(state, humanRep, checkGroundState = True)
			r = 0
			if np.count_nonzero(state) > 0:
				i += 1
				
				x, y = np.where(humanRep == -1)
				flips = len(x)
				print("Iteration ",  i)
				#print("state:\n", state)
				amountOfErrors = env.getAmountOfErrors()
				plaqCount += amountOfErrors
				
				plaq[int(amountOfErrors/2)-1] += 1
				
				
				#numIters = 2*np.sum(i for i in range(0,env.getAmountOfErrors() ))
				#numIters = 5*env.getAmountOfErrors()
				#print("numIters: ", numIters)
				"""
				MWPM = list()
				if amountOfErrors >= 10:
					for index in range(10):
						G = Blossom(state)
						MWPM.append(G.readResult())
				else:
					MWPM = MCGraph(state).getMWPM()
					
				MWPM = [k for k,v in groupby(sorted(MWPM))]
				#print("All MWPM: ", MWPM)
				bestMatch = []
				old_max = 0
				#print("length of MWPM: ", len(MWPM))
				for match in MWPM:
					#print("match: ", match)
					tot, dist = env.chooseMatch(match)
					if tot > old_max:
						bestMatch = match
						old_max = tot
				
				"""
				#print("Best match: ", bestMatch, " comb: ", old_max)
				MWM = Blossom(state).readResult()
				MWM_area, MWM_dist = env.chooseMatch(MWM)
				#print("MWPM: ", MWM, " comb: ", MWM_area, " dist: ", MWM_dist)
				
				MWM_euclid = EBlossom(state).readResult()
				MWM_euclid_area, MWM_euclid_dist = env.chooseMatch(MWM_euclid)
				#print("MWPM Euclidean: ", MWM_euclid, " comb: ", MWM_euclid_area, " dist: ", MWM_euclid_dist)
				
				env2 = Env(state, humanRep, checkGroundState = True)
				for element in MWM:
					error1 = element[0]+1
					error2 = element[1]+1
					
					r1 = env2.blossomCancel(error1, error2)
				if r1 == env2.correctGsR:
					plaqGS[int(amountOfErrors/2)-1] +=1
					
				print("Average number of error plaquettes: ", plaqCount/i)
				for kk in range(12):
					if plaq[kk]>1:
						print("Average GS for " + str((kk+1)*2) + " plaquettes: " + str((plaqGS[kk]/plaq[kk])*100) + " % (" + str(plaq[kk]) + " cases)")
				
				"""
				if MWM_area != old_max:
					#print("Same area: ", MWM_area == old_max)
					for element in bestMatch:
						error1 = element[0]+1
						error2 = element[1]+1
						
						#print("error1: ", error1)
						#print("error2: ", error2)
						r = env.blossomCancel(error1, error2)
					
					env2 = Env(state, humanRep, checkGroundState = True)
					for element in MWM:
						error1 = element[0]+1
						error2 = element[1]+1
						
						r1 = env2.blossomCancel(error1, error2)
					
					if r == env.correctGsR:
						self.X += 1
						#print("CORRECT GROUND STATE")
					#else:
						#print("WRONG GROUND STATE")
					#if r1== env2.correctGsR:
						#print("Vanilla Correct")
					#else:
						#print("Vanilla wrong")
					self.n += 1
					#print("Correct GS: ", self.X / self.n)
				"""
				
			"""
			Eenv = Env(state, humanRep, checkGroundState = True)
			Menv = Env(state, humanRep, checkGroundState = True)
			
			if np.count_nonzero(state) > 0:
				ManhattanBlossom = Blossom(state)
				EuclidianBlossom = EBlossom(state)
				EuclidianMWPM = EuclidianBlossom.readResult()
				ManhattanMWPM = ManhattanBlossom.readResult()
				#print("state:\n", state)
				#print("Eucl: ", EuclidianMWPM)
				#print("Manh: ", ManhattanMWPM)
				if len(EuclidianMWPM) != len(ManhattanMWPM):
					print("state:\n", state)
					print("Eucl: ", EuclidianMWPM)
					print("Manh: ", ManhattanMWPM)
				
				if EuclidianMWPM != ManhattanMWPM:
					#print("Eucl: ", EuclidianMWPM)
					#print("Manh: ", ManhattanMWPM)
					for element in EuclidianMWPM:
						error1 = element[0]
						#print("error1: ", error1)
						error2 = element[1]
						#print("error2: ", error2)
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
					
					if self.n_manhattan == 10000:
						break
					print("Matching: ", self.n_manhattan)
			
			"""

if __name__ == '__main__':
	runBlossom()
