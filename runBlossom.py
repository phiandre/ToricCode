import numpy as np
from Blossom import Blossom
from BlossomEnv import Env

class runBlossom:
	
	def __init__(self):
		self.X = 0
		self.n = 0
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
	
	def run(self):
		comRep = np.load('ToricCodeComputerTest.npy')
		humRep=np.load('ToricCodeHumanTest.npy')
		size = comRep.shape[0]
		
		for i in range(comRep.shape[2]):
			state = np.copy(comRep[:,:,i])
			humanRep = humRep[:,:,i]
			
			state = self.labelState(state,state.shape[0])
			
			env = Env(state, humanRep, checkGroundState = True)
			
			if np.count_nonzero(state) > 0:
				BlossomClass = Blossom(state)
				MWPM = BlossomClass.readResult()
				#print("MWPM: ",MWPM)
				
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
					
			
			

if __name__ == '__main__':
	runBlossom()
