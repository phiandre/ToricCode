import numpy as np
from Blossom import Blossom
from BlossomEnv import BlossomEnv

class runBlossom:
	
	def __init__(self):
		self.run()
	
	def run(self):
		comRep = np.load('ToricCodeComputer.npy')
		humRep=np.load('ToricCodeHuman.npy')
		size = comRep.shape[0]
		
		for i in range(comRep.shape[2]):
			state = comRep[:,:,i]
			humanRep = humRep[:,:,i]
			
			env = BlossomEnv(state, humanRep, checkGroundState = True)
			
			if np.count_nonzero(state) > 0:
				BlossomClass = Blossom(state)
				MWPM = BlossomClass.readResult()
				print("MWPM: ",MWPM)
				print("humRep:\n", humanRep)
				
				#for element in MWPM:
				print("MWPM[2][0]",MWPM[2][0])
				env.cancelErrors(MWPM[2][0],MWPM[2][1])
				print("After corr:\n",env.getHumanState())
					
			
			

if __name__ == '__main__':
	runBlossom()
