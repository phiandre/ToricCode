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
import time

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
	def __init__(self, compState, humanState=np.zeros(0), windowSize = 5, segmentSize = 3, groundState=0, checkGroundState=False, copy=True):
		# Spara viktiga matriser och variabler
		self.checkGroundState = checkGroundState
		if copy:
			self.state = np.copy(compState)
			self.humanState = np.copy(humanState)
		else:
			self.state = compState
			self.humanState = humanState
		self.length = self.state.shape[0]
		self.segmentSize = segmentSize
		self.windowSize = windowSize
		self.groundState = groundState
		# Uppdatera platser där fel finns
		self.updateErrors()
		self.numErrors = 5
		self.stepR = -1
		self.correctGsR = 5
		self.incorrectGsR = -1
		self.elimminationR = 5

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
					return self.correctGsR
				else:
					return self.incorrectGsR
		if amountErrors > len(self.errors):
			return self.elimminationR
		return self.stepR

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

		colmid -= 1
		rowmid -= 1
		lower_x = int((colmid-np.floor(self.windowSize/2)))
		higher_x = int((colmid+np.ceil(self.windowSize/2)))
		lower_y = int((rowmid-np.floor(self.windowSize/2)))
		higher_y = int((rowmid+np.ceil(self.windowSize/2)))
		"""
		print("rowmid", rowmid)
		print("colmid", colmid)
		print("lower_x", lower_x)
		print("higher_x", higher_x)
		print("lower_y", lower_y)
		print("higher_y", higher_y)
		"""
		state = state_[lower_x:higher_x, lower_y:higher_y]
		return state

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
		alone = True
		numerror=self.numErrors
		indexVector = np.zeros(numerror)
		if len(self.errors)==0:
			alone = False

			return 'terminal', alone, indexVector
		else:
			#observation=np.zeros((self.windowSize,self.windowSize, numerror))
			observation=np.zeros((numerror,self.windowSize,self.windowSize))
			added = 0
			indexList = list(range(len(self.getErrors())))

			while( (added != numerror) and len(indexList)>0 ):
				index = indexList.pop(np.random.randint(0,len(indexList)))
				centralizedState = self.centralize(self.errors[index,:])
				if len(np.transpose(np.nonzero(centralizedState))) == 1:
					continue
				else:
					indexVector[added] = index
					#observation[:,:, added] = centralizedState
					observation[added,:,:] = centralizedState
					added += 1
					alone = False
			if added == 0:
				indexList = list(range(len(self.getErrors())))
				while ( (added != numerror) and len(indexList)>0 ):
					index = indexList.pop(np.random.randint(0, len(indexList)))
					centralizedState = self.centralize(self.errors[index, :])
					indexVector[added] = index
					#observation[:, :, added] = centralizedState
					observation[added,:, :] = centralizedState
					added += 1

			#observation = observation[:,:,0:added]
			observation = observation[0:added,:,:,np.newaxis]
			indexVector = indexVector[0:added]
			#print("Observation shape: ",observation.shape)

			return observation, alone, indexVector


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


	def zoomOut(self):
		new_length = int(self.length/self.segmentSize)
		new_state = np.zeros((new_length , new_length))
		for i in range(new_length):
			for j in range(new_length):
				partition = self.state[i*self.segmentSize:(i+1)*self.segmentSize,j*self.segmentSize:(j+1)*self.segmentSize]
				if len(np.transpose(np.nonzero(partition))) != 0:
					new_state[i,j] = 1
		return new_state

	def longMove(self, action, coords):
		xcoord = int(coords[0])
		ycoord = int(coords[1])

		segmentStart_x = self.segmentSize*xcoord
		segmentEnd_x = self.segmentSize*(xcoord+1)
		segmentStart_y = self.segmentSize*ycoord
		segmentEnd_y = self.segmentSize*(ycoord+1)

		partition = self.state[segmentStart_x:segmentEnd_x, segmentStart_y:segmentEnd_y]
		#print("partition\n", partition)
		error_x, error_y = np.where(partition == 1)
		#print("errorx: ", error_x, " errory: ", error_y)
		#error_x += 1
		#error_y += 1
		#partition = self.state[(segmentStart_x-1):(segmentEnd_x+1), (segmentStart_y-1):(segmentEnd_y+1)]

		absolute_x = error_x[0] + segmentStart_x
		absolute_y = error_y[0] + segmentStart_y
		#print("abs_x ", absolute_x, " abs_y", absolute_y)

		absoluteCoords = np.array([absolute_x,absolute_y])

		errorList = self.getErrors().tolist()
		errorIndex = errorList.index(absoluteCoords.tolist())

		steps = 0
		if action == 0:
			while len(np.transpose(np.nonzero(partition))) > 0:
				errorIndex = self.getErrors().tolist().index(list(absoluteCoords))
				r = self.moveError(action,errorIndex)
				steps+=1
				if absoluteCoords[0] != 0:
					absoluteCoords[0] -= 1
				else:
					absoluteCoords[0] = self.length-1

		elif action == 1:
			while len(np.transpose(np.nonzero(partition))) > 0:
				errorIndex = self.getErrors().tolist().index(list(absoluteCoords))
				r = self.moveError(action, errorIndex)
				steps += 1
				if absoluteCoords[0] != self.length - 1:
					absoluteCoords[0] += 1
				else:
					absoluteCoords[0] = 0

		elif action == 2:
			while len(np.transpose(np.nonzero(partition))) > 0:
				errorIndex = self.getErrors().tolist().index(list(absoluteCoords))
				r = self.moveError(action, errorIndex)
				steps += 1
				if absoluteCoords[1] != 0:
					absoluteCoords[1]-= 1
				else:
					absoluteCoords[1] = self.length-1

		elif action == 3:
			while len(np.transpose(np.nonzero(partition))) > 0:
				errorIndex = self.getErrors().tolist().index(list(absoluteCoords))
				r = self.moveError(action, errorIndex)
				steps += 1
				if absoluteCoords[1] != self.length - 1:
					absoluteCoords[1] += 1
				else:
					absoluteCoords[1] = 0
		return steps, r


	def pairErrors(self, coords):
		xcoord = coords[0]
		ycoord = coords[1]


		segmentStart_x = self.segmentSize * xcoord
		segmentEnd_x = self.segmentSize*(xcoord + 1)
		segmentStart_y = self.segmentSize * ycoord
		segmentEnd_y = self.segmentSize * (ycoord + 1)

		partition = self.state[segmentStart_x:segmentEnd_x, segmentStart_y:segmentEnd_y]

		humanPartiton = self.humanState[2*segmentStart_x:2*segmentEnd_x, 2*segmentStart_y:2*segmentEnd_y]


		x, y = np.where(partition == 1)

		if len(x) == 0:
			if len(self.getErrors() == 0):
				if self.evaluateGroundState() == 0:
					r = self.correctGsR
				else:
					r = -1
			else:
				r = -1
			return 0, r

		x1 = x[0]
		y1 = y[0]
		x2 = x[1]
		y2 = y[1]
		ydist = y2-y1
		xdist = x2-x1

		partitionEnv = Env(partition, humanPartiton, checkGroundState=True, copy = False)
		partitionEnv.correctGsR = self.correctGsR
		#partitionEnv = Env(partition, checkGroundState=self.checkGroundState, copy=False)
		r = 0
		print("After")
		if ydist<0:
			for i in range(np.abs(ydist)):
				r = partitionEnv.moveError(2,0)
		else:
			for i in range(np.abs(ydist)):
				r = partitionEnv.moveError(3,0)

		for i in range(np.abs(xdist)):
			r = partitionEnv.moveError(1,0)

		self.updateErrors()
		return int(np.abs(xdist)+np.abs(ydist)), r





if __name__ == '__main__':
	"""
	comRep = np.load('ToricCodeComputer.npy')
	humRep = np.load('ToricCodeHuman.npy')

	state = comRep[:, :, 0]
	human = humRep[:, :, 0]

	print("state\n", state)
	print("human\n", human)
	"""
	state = np.zeros((9,9))
	state[0,5] = 1
	state[5,3] = 1

	env = Env(state)
	print("in \n", env.state)
	zoomedOutState = env.zoomOut()
	zoomedOut = Env(zoomedOutState)
	print("out \n", zoomedOut.state)
	env.longMove(1,np.array([0,1]))

	env.pairErrors(np.array([1,1]))
	print("after\n", env.state)



	

