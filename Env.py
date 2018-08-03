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
	def __init__(self, compState, segmentSize ,humanState=np.zeros(0), groundState=0, checkGroundState=False):
		# Spara viktiga matriser och variabler
		self.checkGroundState = checkGroundState
		self.state = np.copy(compState) 
		self.humanState = np.copy(humanState) 
		
		self.length = self.state.shape[0]
		self.segmentSize = segmentSize
		self.metaLength = int(self.length/self.segmentSize)
		
		self.groundState = groundState
		# Uppdatera platser där fel finns
		self.updateErrors()
		
		self.segments = self.oddSegments()
		
		print("segments:\n", self.segments)
		self.moveSegment(1,5)
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

	"""""""""""""""""""""""""""""""""""""""
	Returnerar matris innehållande felen.
		@return
			numpy: matris innehållande fel.
	"""""""""""""""""""""""""""""""""""""""
	def getErrors(self):					
		return self.errors

	
	def getCoords(self, firstPos, currSeg):
		print("currSeg\n", currSeg)
		coords_x, coords_y = np.where(currSeg == 1)
		coords_x = self.segmentSize*firstPos[0]+coords_x
		coords_y = self.segmentSize*firstPos[1]+coords_y
		return coords_x, coords_y
	
	
	def moveSegment(self, action, segmentIndex):
		firstPos = self.getOddSegments()[segmentIndex,:]
		secondPos = self.getPos(action, firstPos)
		
		#  Uppdatera den gamla positionen
		self.segments[firstPos[0], firstPos[1]] = 0
		# Uppdatera den nya positionen
		if self.segments[secondPos[0], secondPos[1]] == 0:
			self.segments[secondPos[0], secondPos[1]] = 1
		else:
			self.segments[secondPos[0],secondPos[1]] = 0
		notDone = True		
		
		while notDone:
			rowStart = self.segmentSize*firstPos[0]
			rowEnd = (firstPos[0]+1)*self.segmentSize
			colStart = self.segmentSize*firstPos[1]
			colEnd = (firstPos[1]+1)*self.segmentSize
			currSeg = self.state[rowStart:rowEnd, colStart:colEnd]
			print("humanState before:\n", self.humanState)
			
			coords_x, coords_y = self.getCoords(firstPos,currSeg)
			if action == 0: #up
				if firstPos[0] == 0:
					lastRow = self.state[self.metaLength*self.segmentSize-1,(self.segmentSize*firstPos[1]):((firstPos[1]+1)*self.segmentSize)]
				else:
					lastRow = self.state[(self.segmentSize*firstPos[0]-1), (self.segmentSize*firstPos[1]):((firstPos[1]+1)*self.segmentSize) ]
					#nextSeg = np.concatenate(lastRow, currSeg, axis = 0)
					#nextSeg = np.copy(self.state[(self.segmentSize*firstPos[0]-1):((firstPos[0]+1)*self.segmentSize),(self.segmentSize*firstPos[1]):((firstPos[1]+1)*self.segmentSize) ])
				lastRow.shape = (1,lastRow.shape[0])
				targetSeg = np.concatenate((lastRow, np.zeros((self.segmentSize-1,self.segmentSize))), axis = 0)
				update = np.abs(targetSeg - currSeg)
				lastRow[:] = update[0,:]
				currSeg[0:self.segmentSize-1, :] = update[1:,:]  
				currSeg[self.segmentSize-1,:] = 0
				notDone = self.checkOdd(firstPos)
				
				row = 2*coords_x
				col = 2*coords_y+1
				
			elif action == 1: #down
				row = 2*coords_x+2
				col = 2*coords_y+1
				if firstPos[0] == self.metaLength-1:
					firstRow = self.state[0, (self.segmentSize*firstPos[1]):((firstPos[1]+1)*self.segmentSize) ]
					for i in range(row.shape[0]):
						if coords_x[i] == self.length-1:
							row[i] = 0
					#nextSeg = np.concatenate(currSeg, firstRow, axis = 0)
				else:
					firstRow = self.state[((firstPos[0]+1)*self.segmentSize + 1), (self.segmentSize*firstPos[1]):((firstPos[1]+1)*self.segmentSize) ]
					
					#nextSeg = np.copy(self.state[(self.segmentSize*firstPos[0]):((firstPos[0]+1)*self.segmentSize + 1),(self.segmentSize*firstPos[1]):((firstPos[1]+1)*self.segmentSize) ])
				firstRow.shape = (1,firstRow.shape[0])
				targetSeg = np.concatenate( (np.zeros((self.segmentSize-1, self.segmentSize)), firstRow), axis = 0)
				update = np.abs(targetSeg - currSeg)
				firstRow[:] = update[self.segmentSize-1,:]
				currSeg[1:self.segmentSize, :] = update[0:self.segmentSize-1,:] 
				currSeg[0,:] = 0 
				notDone = self.checkOdd(firstPos)
				
			
			elif action == 2: #left
				if firstPos[1] == 0:
					lastCol = self.state[(self.segmentSize*firstPos[0]):((firstPos[0]+1)*self.segmentSize),self.metaLength*self.segmentSize-1]
					#nextSeg = np.concatenate(lastCol, currSeg, axis = 1)
				else:
					lastCol = self.state[(self.segmentSize*firstPos[0]):((firstPos[0]+1)*self.segmentSize),(self.segmentSize*firstPos[1]-1)]
					#nextSeg = np.copy(self.state[(self.segmentSize*firstPos[0]):((firstPos[0]+1)*self.segmentSize),(self.segmentSize*firstPos[1]-1):((firstPos[1]+1)*self.segmentSize) ])
				lastCol.shape = (lastCol.shape[0],1)
				targetSeg = np.concatenate( (lastCol, np.zeros((self.segmentSize, self.segmentSize-1))), axis = 1)
				update = np.abs(targetSeg - currSeg)
				
				
				lastCol[:,0] = update[:,0]
				
				currSeg[:,0:self.segmentSize-1] = update[:,1:]  
				currSeg[:,self.segmentSize-1] = 0
				notDone = self.checkOdd(firstPos)
				row = 2*coords_x+1
				col = 2*coords_y
				
			elif action == 3: #right
				row = 2*coords_x+1
				col = 2*coords_y+2
				if firstPos[1] == self.metaLength-1:
					firstCol = self.state[(self.segmentSize*firstPos[0]):((firstPos[0]+1)*self.segmentSize),0]
					for i in range(row.shape[0]):
						if coords_y[i] == self.length-1:
							col[i] = 0
					#nextSeg = np.concatenate(currSeg, firstCol, axis = 1)
				else:
					firstCol = self.state[(self.segmentSize*firstPos[0]):((firstPos[0]+1)*self.segmentSize),( (firstPos[1]+1)*self.segmentSize)]
					#nextSeg = np.copy(self.state[(self.segmentSize*firstPos[0]):((firstPos[0]+1)*self.segmentSize),(self.segmentSize*firstPos[1]):((firstPos[1]+1)*self.segmentSize + 1)])
				
				firstCol.shape = (firstCol.shape[0],1)
				targetSeg = np.concatenate( (np.zeros((self.segmentSize, self.segmentSize-1)), firstCol), axis = 1)
				update = np.abs(targetSeg - currSeg)
				firstCol[:,0] = update[:,self.segmentSize-1]
				currSeg[:,1:self.segmentSize] = update[:,0:(self.segmentSize-1)]  
				currSeg[:,0] = 0
				notDone = self.checkOdd(firstPos)
			
			for k in range(len(row)):
				self.humanState[row[k],col[k]] *= -1
			print("humanState after:\n", self.humanState)
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
				nextPos[0] = self.metaLength - 1
			else:
				nextPos[0] -= 1  
		if action == 1:
			if nextPos[0] == self.metaLength - 1:
				nextPos[0] = 0
			else:
				nextPos[0] += 1
		if action == 2:
			if nextPos[1] == 0:
				nextPos[1] = self.metaLength - 1
			else:
				nextPos[1] -= 1
		if action == 3:
			if nextPos[1] == self.metaLength - 1:
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
		
	
	def oddSegments(self):
		newA = np.zeros((4,4))
		copiedState = np.copy(self.state)
		for i in range(5):
			for j in range(5):
				B = copiedState[5*i:(5*i+5),5*j:(5*j+5)]
				numErrors = len(np.transpose(np.nonzero(B)))
				if numErrors%2!=0:
					newA[i,j] = 1
		return newA
		
	def getOddSegments(self):
		return np.transpose(np.nonzero(self.segments))
		
	def checkOdd(self, metaIndex):
		region = np.copy(self.state[metaIndex[0]*self.segmentSize:(metaIndex[0]+1)*self.segmentSize, metaIndex[1]*self.segmentSize:(metaIndex[1] +1)*self.segmentSize])
		numErrors = len(np.transpose(np.nonzero(region)))
		return numErrors%2 != 0
		
	

if __name__ == '__main__':
	compState = np.zeros((20,20))
	compState[0,1] = 1
	compState[0,2] = 1
	compState[3,4] = 1
	compState[18,2] = 1
	compState[18,3] = 1
	compState[17,1] = 1
	np.set_printoptions(threshold=np.nan, linewidth=300)
	comRep = np.load('ToricCodeComputer.npy')
	humRep=np.load('ToricCodeHuman.npy')
	#print("humRep\n", humRep[:,:,0])
	env = Env(comRep[:,:,0],5,humRep[:,:,0])
