import numpy as np
import random
import pickle


class Generate:
	
	""""
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
	"""
	def generateData(size, numFlips):
		humanRepresentation, computerRepresentation = Generate.initialize(size)
		
		for iteration in range(0,numFlips):
			rowIndex, columnIndex = Generate.flipRandomIndex(2*size)
			while (humanRepresentation[(rowIndex, columnIndex)]== -1):
				rowIndex, columnIndex = Generate.flipRandomIndex(2*size)
			
			humanRepresentation[(rowIndex, columnIndex)] = -1*humanRepresentation[(rowIndex, columnIndex)];
			
			computerRepresentation = Generate.updateComputerRepresentation(rowIndex, columnIndex, size, computerRepresentation)
		computerRepresentation = abs((computerRepresentation-1)/(2))
		#print(computerRepresentation)
		
		return humanRepresentation, computerRepresentation
		
	"""
		initialize skapar de ursprungliga representationerna av 
		humanRepresentation och computerRepresentation, kallas på i GenerateData.
		@param
			size: storleken på gittret, samma som i GenerateData ovan.
		@return
			humanRepresentation: den enklaste formen av grundtillstånd
								för humanRepresentation.
			computerRepresentation: den enklaste formen av grundtillstånd
								för computerRepresentation.
	"""
	def initialize(size):
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
		computerRepresentation = np.ones((size,size))
		
		
		return humanRepresentation, computerRepresentation
		

	"""
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
	
	"""

	def updateComputerRepresentation(humanRowIndex, humanColumnIndex, size, rep):
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
			
		return computerRepresentation
		
	
	"""
		flipRandomIndex väljer ut ett random index i humanRepresentation
		som ska flippas, kallas på i GenerateData.
		
		@param
			size: storleken på gittret
		
		@return
			rowIndex: radindex i humanRepresentation för spinnet som 
			ska flippas
			columnIndex: kolonnindex i humanRepresentation för spinnet
			som ska flippas
	"""
	
	def flipRandomIndex(size):
		
		rowIndex = random.randint(0,size-1)
		columnIndex = random.randint(0,size-1)
		
		while rowIndex % 2 == 1 and columnIndex % 2 == 1:
			columnIndex = random.randint(0,size-1)
		while rowIndex % 2 == 0 and columnIndex % 2 == 0:
			columnIndex = random.randint(0,size-1)
		return rowIndex , columnIndex
	
	
	"""
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
	"""
	
	def saveToFile(human, computer):
		np.save('ToricCodeHuman',human)
		np.save('ToricCodeComputer', computer)
	
			
if __name__ == '__main__':
	size = 5 #Storlek på gittret
	numFlips = 6 #Antalet spin som ska flippas
	
	numGenerations = 100 #Antalet datapunkter som ska skapas
	
	tmpHuman = np.zeros((size*2,size*2,numGenerations))
	tmpComputer = np.zeros((size,size,numGenerations))
	for i in range(numGenerations):
		human, computer = Generate.generateData(size,numFlips)
		tmpHuman[:,:,i] = human
		tmpComputer[:,:,i] = computer
	
	Generate.saveToFile(tmpHuman, tmpComputer)
