##########
# Import #
##########
import math
import numpy as np

###############
# Klassen Env #
###############

class BlossomEnv:
	
	"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
	Konstruktor för klass Env.
		@param
			compState: tar in initial statematris i numpy.
			humanState: tar in spin-matris (för reward).
			groundState: grundtillståndet, värde från 0 till 3.
			checkGroundState: beroende på om man vill kolla grundtillstånd.
	"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
	def __init__(self, compState, humanState=np.zeros(0), groundState=0, checkGroundState=False):
		# Spara viktiga matriser och variabler
		self.checkGroundState = checkGroundState
		self.state = np.copy(compState) 
		self.humanState = np.copy(humanState) 
		self.length = self.state.shape[0]
		self.groundState = groundState
		# Uppdatera platser där fel finns
		self.updateErrors()
		
		self.stepR = -1
		self.correctGsR = 5
		self.incorrectGsR = -1
		self.elimminationR = -1
		
		
		"""""""""""""""""""""""""""""""""""""""""""""""""""
	Hittar alla fel och uppdaterar matrisen errors där
	felens koordinater finns uppradade.
	"""""""""""""""""""""""""""""""""""""""""""""""""""
	def updateErrors(self):
		self.errors = np.transpose(np.nonzero(self.state))
		if(len(self.errors)==0):
			if self.groundState == self.evaluateGroundState():
				return 1
			else:
				return -1

	"""""""""""""""""""""""""""""""""""""""
	Returnerar matris innehållande felen.
		@return
			numpy: matris innehållande fel.
	"""""""""""""""""""""""""""""""""""""""
	def getErrors(self):					
		return self.errors
		
	
	def cancelErrors(self, error1, error2):
		colSteps = error2[0]-error1[0]
		rowSteps = error2[1]-error1[1]
		
		for i in range(abs(colSteps)):
			if colSteps > 0:
				self.humanState[(2*(error1[0]+1)-1),(2*(error1[1]+1)-1)+2*i+1] *= -1
			else:
				self.humanState[(2*(error1[0]+1)-1),(2*(error1[1]+1)-1)-2*i-1] *= -1
		
		for i in range(abs(rowSteps)):
			if rowSteps > 0:
				self.humanState[(2*(error1[0]+1)-1)+2*i+1,(2*(error1[1]+1)-1)+2*colSteps] *= -1
			else:
				self.humanState[(2*(error1[0]+1)-1)-2*i-1,(2*(error1[1]+1)-1)-2*colSteps] *= -1
		
		self.updateErrors()
		
				
	""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
	Avgör vilket grundtillstånd vi befinner oss i
		@precondition
			kräver att vi har en felfri tillståndsmatris.
		@return
			int: grundtillstånd som vi har just nu.

	Grundtillstånd: 0 (inga icketriviala loopar)
					1 (vertikal icketrivial loop)
					2 (horisontell icketrivial loop)
					3 (vertikal + horisontell icketrivial loop)
	"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
	def evaluateGroundState(self):
		yProd = 1
		xProd = 1
		groundState = 0
		for i in range(self.length):
			yProd = yProd * self.humanState[2*i+1, 0]
			xProd = xProd * self.humanState[0, 2*i+1]
		if (yProd == -1):
			groundState += 1
		if (xProd == -1):
			groundState += 2
		return groundState
		
	def getHumanState(self):
		return self.humanState
		
		
