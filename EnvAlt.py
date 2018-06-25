"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

	Klassen Env skapar ett objekt som är en matris för kvantdatan.
	Den kan uppdateras genom att flytta ett specifikt fel (error) i
	matrisen genom att utföra en förflyttning (action).

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

##########
# Import #
##########
import math
import numpy as np

###############
# Klassen Env #
###############
class Env:

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
		self.labelErrors()
		print(self.state)
		print(' ')
		print(self.humanState)

	"""""""""""""""""""""""""""""""""""""""""""""""""""
	Hittar alla fel och uppdaterar matrisen errors där
	felens koordinater finns uppradade.
	"""""""""""""""""""""""""""""""""""""""""""""""""""
	def updateErrors(self):
		self.errors = np.transpose(np.nonzero(self.state))

	"""""""""""""""""""""""""""""""""""""""
	Returnerar matris innehållande felen.
		@return
			numpy: matris innehållande fel.
	"""""""""""""""""""""""""""""""""""""""
	def getErrors(self):					
		return self.errors

	""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
	Flyttar errors, och släcker ut som två errors möter varandra.
	Actions följer: [u = 0, d = 1, l = 2, r = 3]
		@param
			action: rörelse som vi vill utföra.
			errorIndex: index till fel som vi vill flytta.
		@return
			int: reward, 10 för att ta bort, -1 för ingen skillnad.
	"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
	def moveError(self, action, errorIndex):
		# Kolla antal errors innan
		amountErrors = len(self.errors)
		# Positionen för felet som skall flyttas
		firstPos = self.errors[errorIndex, :]
		# Nya positionen för felet givet action och position
		secondPos = self.getPos(action, firstPos)
		
		# Uppdatera humanState
		if self.checkGroundState:
			# Positionen för felet i humanState
			firstHumPos=2*firstPos+1
			# Positionen för felets nya plats i humanState
			secondHumPos=2*secondPos+1
			if action==0 and firstPos[0]==0:
				vertexPos = [0, firstHumPos[1]]
			elif action==1 and firstPos[0]==self.length - 1:
				vertexPos = [0, firstHumPos[1]]
			elif action==2 and firstPos[1]==0:
				vertexPos = [firstHumPos[0], 0]
			elif action==3 and firstPos[1]==self.length - 1:
				vertexPos = [firstHumPos[0], 0]
			else:
				vertexPos = 1/2 * (firstHumPos + secondHumPos)
				vertexPos = vertexPos.astype(int)
			self.humanState[vertexPos[0], vertexPos[1]] *= -1
		
		#  Uppdatera den gamla positionen
		self.state[firstPos[0], firstPos[1]] = 0
		# Uppdatera den nya positionen
		if self.state[secondPos[0], secondPos[1]] == 0:
			self.state[secondPos[0], secondPos[1]] = 1
		else:
			self.state[secondPos[0],secondPos[1]] = 0
		# Kolla igenom igen var fel finns
		self.updateErrors()
		# I fallet att vi är klara, se om vi har bevarat grundtillstånd
		if self.checkGroundState:
			if len(self.errors) == 0:
				if (self.evaluateGroundState() == self.groundState):
					return 10
				else:
					return -10
		if amountErrors > len(self.errors):
			return 10
		return -1
	
	def labelErrors(self):
		errors = self.getErrors()
		for i in range(len(errors)):
			pos = errors[i,:]
			self.state[pos[0],pos[1]] = i+1
		
	def cancelTuple(self,tuple,rowSteps,colSteps):
		errors = self.getErrors()
		error1 = tuple[0]
		error2 = tuple[1]
		pos1=errors[(error1-1),:]
		
		for i in range(abs(colSteps)):
			if colSteps > 0:
				self.humanState[(2*(pos1[0]+1)-1),(2*(pos1[1]+1)-1)+2*i+1] *= -1
			else:
				self.humanState[(2*(pos1[0]+1)-1),(2*(pos1[1]+1)-1)-2*i-1] *= -1
		for i in range(abs(rowSteps)):
			if rowSteps > 0:
				self.humanState[(2*(pos1[0]+1)-1)+2*i+1,(2*(pos1[1]+1)-1)+2*colSteps] *= -1
			else:
				self.humanState[(2*(pos1[0]+1)-1)-2*i-1,(2*(pos1[1]+1)-1)+2*colSteps] *= -1
		print(' ')
		print(self.humanState)
	""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
	Flyttar errors, och släcker ut som två errors möter varandra.
	Actions följer: [u = 0, d = 1, l = 2, r = 3]
		@param
			error: index till felet vi vill centralisera.
		@return
			numpy: state centrerat kring hänvisat error.
	"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""		
	def centralize(self, error):
		# state är matrisen som karaktäriserar tillståndet
		# error är koordinaterna för felet
		state_=np.concatenate((self.state[:,error[1]:],self.state[:,0:error[1]]),1)
		state_=np.concatenate((state_[error[0]:,:],state_[0:error[0],:]),0)
		rowmid=int(np.ceil(self.state.shape[0]/2))
		colmid=int(np.ceil(self.state.shape[1]/2))
		state_=np.concatenate((state_[:,colmid:],state_[:,0:colmid]),1)
		state_=np.concatenate((state_[rowmid:,:],state_[0:rowmid,:]),0)
		
		return state_
	
	""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
	Returnerar positionen efter att ha rört sig i en viss riktning.
		@param
			action: den associerade riktningsrörelsen.
			position: positionen vi står vid innan vi flyttar oss.
		@return
			numpy: koordinater för nya positionen.
	"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
	def getPos(self, action, position):
		# Kopiera tidigare position så vi får ny pekare
		nextPos = np.array(position, copy=True)
		# Beroende på action väljs steg
		if action == 0:
			if nextPos[0] == 0:
				nextPos[0] = self.length - 1
			else:
				nextPos[0] -= 1  
		if action == 1:
			if nextPos[0] == self.length - 1:
				nextPos[0] = 0
			else:
				nextPos[0] += 1
		if action == 2:
			if nextPos[1] == 0:
				nextPos[1] = self.length - 1
			else:
				nextPos[1] -= 1
		if action == 3:
			if nextPos[1] == self.length - 1:
				nextPos[1] = 0
			else:
				nextPos[1] += 1
		# Returnera nya positionen för felet
		return nextPos

	""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
	Skapa en 3D-matris där plaketter är samma tillstånd som state
	fast centraliserad kring alla olika fel, i samma ordning som
	felen är ordnade i errors-vektorn.
		@return
			numpy: matrisen som beskrivs ovan.
	"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
	def getObservation(self):
		if len(self.errors)==0:
			return 'terminal'
		else:
			numerror=self.errors.shape[0]
			observation=np.zeros((self.length,self.length,numerror))
			for i in range(numerror):
				observation[:,:,i]=self.centralize(self.errors[i,:])
			return observation

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

	"""""""""""""""""""""""""""""""""""""""""""""
	Kopierar ett Env-objekt och returnera det.
		@return
			Env: en kopia av Env-objektet.
	"""""""""""""""""""""""""""""""""""""""""""""
	def copy(self):
		# Kopiera numpys så vi inte får problem
		copyState = np.copy(self.state)
		copyHuman = np.copy(self.humanState)
		# Instansiera en kopia av samma objekt
		copyEnv = Env(copyState, copyHuman, self.groundState, self.checkGroundState)
		# Returnera kopia
		return copyEnv

