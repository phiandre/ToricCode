import numpy as np
import math

class Tweaker:
	
	def __init__(self):

		""" Iterationer
				och
			Övrigt för
		   Datagenerering
		"""
			
	
		# Antal iterationer som ska tränas
		self.trainingIterations = 500
		
		# Antal iterationer som ska testas
		self.testIterations = 500
		
		
		""" Belöningsparametrar"""
		
		# Ta hänsyn till grundtillstånd
		self.checkGS = False
		
		# Använd växande belöning för grundtillstånd
		self.GSgrowth = True
		
		# Belöning för rätt grundtillstånd (om växande: slutgiltig belöning)
		self.correctGsR = 5
		
		# Belöning för fel grundtillstånd
		self.incorrectGsR = -1
		
		# Belöning för steg
		self.stepR = -1		
		
		# växande belöning för korrekt grundtillstånd enligt formeln
		# R_GS = A * tanh(w(x+b))+ B 
		
		# Se MATLAB-fil för parametrarnas effekt
		
		# Kurvans form oberoende av antal iterationer
		self.groundStateShape = False
		
		self.AGS = 0.5 * self.correctGsR
		self.BGS = self.AGS
		if self.groundStateShape:
			self.wGS = math.pi / (0.275 * self.trainingIterations)
			self.bGS = 0.39 * self.trainingIterations
		else:
			self.wGS = math.pi / 55000
			self.bGS = 78000
		
		""" Epsilon decay """
		
		# Använd avtagande epsilon
		self.epsilonDecay = True
		
		# Värde på epsilon (endast relevant om ej avtagande)
		self.epsilon = 0.1
		
		# Epsilonkurvans form oberoende av antal iterationer
		self.epsilonShape = False
		
		self.alpha = -0.7
		
		if self.epsilonShape:
			self.k = self.trainingIterations / 10
		else:
			self.k = 20000
		
		""" Error rate """
		
		# Använd växande felfrekvens
		self.errorGrowth = False
		
		# Felfrekvenskurvans form oberoende av antal iterationer
		self.errorShape = False
		
		# Felfrekvens (om växande: slutgiltig felfrekvens)
		self.Pe = 0.1
		
		# Initial felfrekvens (endast relevant om växande)
		self.Pei = 0.04
		
		# Felfrekvens för testdata
		self.PeTest = 0.12
		# Växande sannolikhet enligt formeln
		# P_e = A * tanh(w(x+b))+ B 
		
		# Se MATLAB-fil för parametrarnas effekt
		
		self.AE = 0.5 * self.Pe - 0.5 * self.Pei
		self.BE = 0.5 * self.Pe + 0.5 * self.Pei
		
		if self.errorShape:
			self.wE = math.pi / (0.125*self.trainingIterations)
			self.bE = -0.13 * self.trainingIterations
		else:
			self.wE = math.pi/25000
			self.bE = -26000
		
	

if __name__ == '__main__':
	
	tweak = Tweaker()
	
	np.save("Tweaks/trainingIterations.npy",tweak.trainingIterations)
	np.save("Tweaks/testIterations.npy",tweak.testIterations)
	np.save("Tweaks/checkGS.npy",tweak.checkGS)
	np.save("Tweaks/GSgrowth.npy",tweak.GSgrowth)
	np.save("Tweaks/correctGsR.npy",tweak.correctGsR)
	np.save("Tweaks/incorrectGsR.npy",tweak.incorrectGsR)
	np.save("Tweaks/stepR.npy",tweak.stepR)
	np.save("Tweaks/groundStateShape.npy",tweak.groundStateShape)
	np.save("Tweaks/AGS.npy",tweak.AGS)
	np.save("Tweaks/BGS.npy",tweak.BGS)
	np.save("Tweaks/wGS.npy",tweak.wGS)
	np.save("Tweaks/bGS.npy",tweak.bGS)
	np.save("Tweaks/epsilonDecay.npy",tweak.epsilonDecay)
	np.save("Tweaks/epsilon.npy",tweak.epsilon)
	np.save("Tweaks/epsilonShape.npy",tweak.epsilonShape)
	np.save("Tweaks/alpha.npy",tweak.alpha)
	np.save("Tweaks/k.npy",tweak.k)
	np.save("Tweaks/errorGrowth.npy",tweak.errorGrowth)
	np.save("Tweaks/errorShape.npy",tweak.errorShape)
	np.save("Tweaks/Pe.npy",tweak.Pe)
	np.save("Tweaks/Pei.npy",tweak.Pei)
	np.save("Tweaks/PeTest.npy",tweak.PeTest)
	np.save("Tweaks/AE.npy",tweak.AE)
	np.save("Tweaks/BEcap.npy",tweak.BE)
	np.save("Tweaks/wE.npy",tweak.wE)
	np.save("Tweaks/bE.npy",tweak.bE)
	np.save("Tweaks/trainingIterations.npy",tweak.trainingIterations)
	np.save("Tweaks/trainingIterations.npy",tweak.trainingIterations)
	np.save("Tweaks/trainingIterations.npy",tweak.trainingIterations)
	np.save("Tweaks/trainingIterations.npy",tweak.trainingIterations)
	np.save("Tweaks/trainingIterations.npy",tweak.trainingIterations)
	np.save("Tweaks/trainingIterations.npy",tweak.trainingIterations)
	np.save("Tweaks/trainingIterations.npy",tweak.trainingIterations)
	np.save("Tweaks/trainingIterations.npy",tweak.trainingIterations)
	np.save("Tweaks/trainingIterations.npy",tweak.trainingIterations)
	np.save("Tweaks/trainingIterations.npy",tweak.trainingIterations)
	np.save("Tweaks/trainingIterations.npy",tweak.trainingIterations)
