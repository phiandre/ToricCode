import numpy as np
import random
import pickle
import time
from Blossom import Blossom

class Generate:
	
	"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
		generateData är huvudmetoden. 
		@param
			size: antalet plaketter längs en sida i gittret. Det blir kvadratiskt
			numFlips: antalet spinflips som ska göras. Finns sannolikhet för att
					spinn flippas så att fel tar ut varandra, alltså kan antalet fel som
					genereras med samma värde på denna parameter variera.
		@return
			humanRepresentation: en representation likt de som gjordes i tidigare delen av
								projektet med isingmodellen. Finns en 0 för varje vertex och 
								varje plaketts mittpunkt. Denna representation kommer inte 
								programmet att använda sig av.
			computerRepresentation: en representation som är en matris av storleken size x size,
								beståendes av ettor och nollor. Ettorna betyder att det finns 
								ett fel i plaketten och nollor betyder att det inte finns 
								något fel.
	"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
	def generateData(self, size, errorProbability, longDistance = False, moveErrorDistance = 5):
		
		comRep = np.ones((size,size))
		
		humanRepresentation, computerRepresentation = self.initialize(size, errorProbability, comRep)
		computerRepresentation = abs((computerRepresentation-1)/(2))
		
		
		return humanRepresentation, computerRepresentation
		
	"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
		initialize skapar de ursprungliga representationerna av 
		humanRepresentation och computerRepresentation, kallas på i GenerateData.
		@param
			size: storleken på gittret, samma som i GenerateData ovan.
		@return
			humanRepresentation: den enklaste formen av grundtillstånd
								för humanRepresentation.
			computerRepresentation: den enklaste formen av grundtillstånd
								för computerRepresentation.
	"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
	def initialize(self, size, errorProbability, computerRep):
		humanRepresentation = np.zeros((2*size,2*size))
		comRep = computerRep
		for i in range(0,2*size):
			if i%2==0:
				for j in range(0,2*size):
					if j%2==1:
						if np.random.uniform() < errorProbability:
							humanRepresentation[i,j] = -1
							comRep = self.updateComputerRepresentation(i,j,size,comRep)
						else:
							humanRepresentation[i,j] = 1
					
			else:
				for j in range(0,2*size):
					if j%2==0:
						if np.random.uniform() < errorProbability:
							humanRepresentation[i,j] = -1
							comRep = self.updateComputerRepresentation(i,j,size,comRep)
						else:
							humanRepresentation[i,j] = 1
		
		return humanRepresentation, comRep
		

	"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
		updateComputerRepresentation anropas när ett spinn har flippats för att 
		uppdatera computerRepresentation. Att ändra detta i humanRepresentation 
		är trivialt, men i computerRepresentation måste man ta hänsyn till att 
		fel kan släckas ut och att spinn kan flippas längst ut på kanterna, kallas på i GenerateData. 
		
		@param
			humanRowIndex: radindexet i humanRepresentation för indexet 
					som flippades
					
			humanColumnIndex: kolonnindexet i humanRepresentation för indexet
					som flippades.
			
			size: storleken på gittret
			rep: den tidigare versionen av computerRepresentation
		
		@return
			computerRepresentation: den uppdaterade versionen av 
							computerRepresentation 
	
	"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

	def updateComputerRepresentation(self, humanRowIndex, humanColumnIndex, size, rep):
		computerRepresentation = rep
		rowIndex = int(np.floor(humanRowIndex / 2))
		columnIndex = int(np.floor(humanColumnIndex / 2))
		if humanRowIndex == 0:
			computerRepresentation[(rowIndex, columnIndex)] = (-1)*computerRepresentation[(rowIndex, columnIndex)]
			computerRepresentation[(size-1, columnIndex)] = (-1)*computerRepresentation[(size-1, columnIndex)]
		elif humanColumnIndex == 0:
			computerRepresentation[(rowIndex, columnIndex)] = (-1)*computerRepresentation[(rowIndex, columnIndex)]
			computerRepresentation[(rowIndex, size-1)] = (-1)*computerRepresentation[(rowIndex, size-1)]
		elif humanRowIndex % 2 == 1:
			computerRepresentation[(rowIndex,columnIndex)] = (-1)*computerRepresentation[(rowIndex,columnIndex)]
			computerRepresentation[(rowIndex,columnIndex-1)] = (-1)*computerRepresentation[(rowIndex,columnIndex-1)]
		else:
			computerRepresentation[(rowIndex,columnIndex)] = (-1)*computerRepresentation[(rowIndex,columnIndex)]
			computerRepresentation[(rowIndex-1,columnIndex)] = (-1)*computerRepresentation[(rowIndex-1,columnIndex)]
		
		# TODO Ska vi inte bara ha 0:or och 1:or som representerar icke-fel och fel? (Tänker på minustecknen) 
		# Behöver vi inte kolla andra gränsen, dvs då tex humanRowIndex = size?
		return computerRepresentation
		
	
	"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
		flipRandomIndex väljer ut ett random index i humanRepresentation
		som ska flippas, kallas på i GenerateData.
		
		@param
			size: storleken på gittret
		
		@return
			rowIndex: radindex i humanRepresentation för spinnet som 
			ska flippas
			columnIndex: kolonnindex i humanRepresentation för spinnet
			som ska flippas
	"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
	
	def flipRandomIndex(self,size):
		
		rowIndex = random.randint(0,size-1)
		columnIndex = random.randint(0,size-1)
		
		while rowIndex % 2 == 1 and columnIndex % 2 == 1:
			columnIndex = random.randint(0,size-1)
		while rowIndex % 2 == 0 and columnIndex % 2 == 0:
			columnIndex = random.randint(0,size-1)
		return rowIndex , columnIndex
	
	
	def adjacentIndex(self,size,rowIndex, columnIndex):
		adjRow = random.randint(-1,1)
		adjCol = random.randint(-1,1)
		
		while adjRow == 0:
			adjRow = random.randint(-1,1)
		while adjCol == 0:
			adjCol = random.randint(-1,1)
		
		rowIndex = rowIndex + adjRow
		columnIndex = columnIndex + adjCol
		
		if rowIndex == 2*size:
			rowIndex = 0
		if rowIndex == -1:
			rowIndex = 2*size-1
		if columnIndex == 2*size:
			columnIndex = 0
		if columnIndex == -1:
			columnIndex = 2*size-1
		
		
		return rowIndex, columnIndex
			
	"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
	def saveToFile(human, computer):
		open('Datafiles/ToricCodeComputer.txt', 'w').close()
		open('Datafiles/ToricCodeHuman.txt', 'w').close()
		with open("Datafiles/ToricCodeComputer.txt","a+") as f:
			
			for i in range(computer.shape[2]):
				f.write(str(computer[:,:,i]))
				f.write("\n")
	
		with open("Datafiles/ToricCodeHuman.txt","a+") as f:
			
			for i in range(human.shape[2]):
				f.write(str(human[:,:,i]))
				f.write("\n")
	"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
	
	def saveToFile(self, human, computer, humanTest, computerTest):
		np.save('ToricCodeHuman',human)
		np.save('ToricCodeComputer', computer)
		np.save('ToricCodeHumanTest',humanTest)
		np.save('ToricCodeComputerTest', computerTest)

			
if __name__ == '__main__':
	size = 15 #Storlek på gittret
	numGenerations = np.load("Tweaks/trainingIterations.npy") # Antalet träningsfall som ska skapas
	testGenerations = np.load("Tweaks/testIterations.npy") # Antalet testfall som ska skapas
	testProb = np.load("Tweaks/PeTest.npy") # error rate för ***testdata***
	Pe = np.load("Tweaks/Pe.npy")
	errorProb = Pe
	Pei = np.load("Tweaks/Pei.npy")
	AE = np.load("Tweaks/AE.npy")
	BE = np.load("Tweaks/BEcap.npy")
	wE = np.load("Tweaks/wE.npy")
	bE = np.load("Tweaks/bE.npy")
	errorGrowth = np.load("Tweaks/errorGrowth.npy")

	generator = Generate()
	#Skapar träningsdata
	tmpHuman = np.zeros((size*2,size*2,numGenerations))
	tmpComputer = np.zeros((size,size,numGenerations))
	for i in range(numGenerations):
		if errorGrowth:
			errorProb = AE * np.tanh(wE*(i+1+bE))+BE
		human, computer = generator.generateData(size,errorProb, False)
		tmpHuman[:,:,i] = human
		tmpComputer[:,:,i] = computer

	errorProb = testProb
	#Skapar testdata
	tmpHumanTest = np.zeros((size*2,size*2,testGenerations))
	tmpComputerTest = np.zeros((size,size,testGenerations))
	for i in range(testGenerations):
		humanTest, computerTest = generator.generateData(size,errorProb, False)
		tmpHumanTest[:,:,i] = humanTest
		tmpComputerTest[:,:,i] = computerTest

	"""
	for i in range(numGenerations):
		label = 1
		labeltest = 1
		for j in range(size):
			for k in range(size):
	
				if tmpComputer[:,:,i][j,k] == 1:
					tmpComputer[:,:,i][j,k] = label
					label +=1
	
	for i in range(testGenerations):
		labeltest = 1
		for j in range(size):
			for k in range(size):
				if tmpComputerTest[:,:,i][j,k] == 1:
					tmpComputerTest[:,:,i][j,k] = labeltest
					labeltest += 1
	
	"""
	generator.saveToFile(tmpHuman, tmpComputer, tmpHumanTest, tmpComputerTest)
